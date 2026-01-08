import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import datetime

# --- KONFIGURACIJA ---
st.set_page_config(page_title="AI Potovalni Agent", layout="wide", page_icon="✈️")

# --- BARVNA LESTVICA ---
CUSTOM_COLOR_SCALE = [
    (0.0, "#77DD77"),  # Pastel Green
    (0.5, "#FDFD96"),  # Pastel Yellow
    (1.0, "#FFB347")   # Pastel Orange
]

# --- FUNKCIJE ---
@st.cache_data
def load_data():
    data = {}
    train_files = ['Data_Train_Cleaned_EUR.csv', 'Data_Train_Cleaned.csv']
    for f in train_files:
        try:
            data['train'] = pd.read_csv(f)
            break
        except FileNotFoundError:
            pass
    test_files = ['Test_set_Cleaned_EUR.csv', 'Test_set.xlsx']
    for f in test_files:
        try:
            if f.endswith('.xlsx'):
                data['test'] = pd.read_excel(f)
            else:
                data['test'] = pd.read_csv(f)
            break
        except FileNotFoundError:
            pass     
    return data

@st.cache_resource
def train_model(df):
    df_train = df.copy()
    encoders = {}
    cats = ['Airline', 'Source', 'Destination', 'Additional_Info']
    
    for c in cats:
        if c not in df_train.columns: return None, None
        le = LabelEncoder()
        df_train[c] = le.fit_transform(df_train[c].astype(str))
        encoders[c] = le

    if 'Price' not in df_train.columns: return None, None
    
    # Odstranimo Route iz učnih podatkov
    X = df_train.drop('Price', axis=1)
    if 'Route' in X.columns:
        X = X.drop('Route', axis=1)
        
    y = np.log1p(df_train['Price'])
    
    model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, encoders

def safe_encode(encoder, val):
    val = str(val)
    return encoder.transform([val])[0] if val in encoder.classes_ else encoder.transform([encoder.classes_[0]])[0]

def predict_row_price(model, encoders, row):
    try:
        input_data = {
            'Airline': safe_encode(encoders['Airline'], row['Airline']),
            'Source': safe_encode(encoders['Source'], row['Source']),
            'Destination': safe_encode(encoders['Destination'], row['Destination']),
            'Total_Stops': row['Total_Stops'],
            'Additional_Info': safe_encode(encoders['Additional_Info'], row.get('Additional_Info', 'No info')),
            'Journey_Day': row['Journey_Day'], 'Journey_Month': row['Journey_Month'], 'Is_Weekend': row['Is_Weekend'],
            'Dep_Hour': row['Dep_Hour'], 'Dep_Min': row['Dep_Min'], 
            'Arrival_Hour': row['Arrival_Hour'], 'Arrival_Min': row['Arrival_Min'],
            'Duration_Total_Mins': row['Duration_Total_Mins']
        }
        df_row = pd.DataFrame([input_data])
        for c in set(model.feature_names_in_) - set(df_row.columns): df_row[c] = 0
        df_row = df_row[model.feature_names_in_]
        return np.expm1(model.predict(df_row))[0]
    except:
        return 0.0

def predict_manual(model, encoders, source, dest, airline, stops, date, avg_dur):
    row = {
        'Airline': airline, 'Source': source, 'Destination': dest,
        'Total_Stops': stops, 'Additional_Info': 'No info',
        'Journey_Day': date.day, 'Journey_Month': date.month,
        'Is_Weekend': 1 if date.weekday() >= 5 else 0,
        'Dep_Hour': 10, 'Dep_Min': 0, 'Arrival_Hour': 14, 'Arrival_Min': 0,
        'Duration_Total_Mins': avg_dur
    }
    return predict_row_price(model, encoders, row)

def format_stops(n):
    if n == 0: return "Neposredno"
    if n == 1: return "1 prestop"
    return f"{n} prestopov"

def format_duration(mins):
    h = int(mins // 60)
    m = int(mins % 60)
    return f"{h}h {m}m"

def get_likely_stopover(df, source, dest, airline, stops):
    if 'Route' not in df.columns: return "-"
    if stops == 0: return "-"
    
    subset = df[(df['Source'] == source) & (df['Destination'] == dest) & (df['Airline'] == airline) & (df['Total_Stops'] == stops)]
    if subset.empty: return "Neznano"
    
    most_common_route = subset['Route'].mode()
    if most_common_route.empty: return "Neznano"
    
    parts = most_common_route[0].split('→')
    parts = [p.strip() for p in parts]
    if len(parts) >= 3:
        return ", ".join(parts[1:-1])
    return "-"

def get_segment_offers(model, encoders, df, source, dest, date, max_stops_allowed):
    route_df = df[(df['Source'] == source) & (df['Destination'] == dest)]
    if route_df.empty: return pd.DataFrame()

    offers = []
    for airline in route_df['Airline'].unique():
        airline_data = route_df[route_df['Airline'] == airline]
        if airline_data.empty: continue
        
        best_price = float('inf')
        best_stops = 0
        best_dur = 0
        found = False
        
        for s in range(max_stops_allowed + 1):
            subset_stops = airline_data[airline_data['Total_Stops'] == s]
            if not subset_stops.empty:
                avg_dur = subset_stops['Duration_Total_Mins'].median()
            else:
                continue 

            price = predict_manual(model, encoders, source, dest, airline, s, date, avg_dur)
            
            if price > 0 and price < best_price:
                best_price = price
                best_stops = s
                best_dur = avg_dur
                found = True
        
        if found:
            stop_location = get_likely_stopover(df, source, dest, airline, best_stops)
            offers.append({
                'Družba': airline, 
                'Cena': best_price, 
                'Št. prestopov': format_stops(best_stops),
                'Verjeten postanek': stop_location,
                'Trajanje': format_duration(best_dur),
                '_raw_duration': best_dur
            })
            
    return pd.DataFrame(offers)

def get_best_daily_prices(model, encoders, df, source, dest, start_date, max_stops_allowed, days=7):
    route_df = df[(df['Source'] == source) & (df['Destination'] == dest)]
    if route_df.empty: return None

    airlines = route_df['Airline'].unique()
    results = {'date': [], 'price': [], 'airline': [], 'stops': []}
    
    for i in range(days):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%d.%m.")
        daily_best_price = float('inf')
        daily_best_airline = "-"
        daily_stops = 0
        
        for airline in airlines:
            airline_data = route_df[route_df['Airline'] == airline]
            if airline_data.empty: continue
            
            for s in range(max_stops_allowed + 1):
                subset_stops = airline_data[airline_data['Total_Stops'] == s]
                if not subset_stops.empty:
                    avg_dur = subset_stops['Duration_Total_Mins'].median()
                else:
                    continue
                
                p = predict_manual(model, encoders, source, dest, airline, s, current_date, avg_dur)
                if p > 0 and p < daily_best_price:
                    daily_best_price = p
                    daily_best_airline = airline
                    daily_stops = s
        
        if daily_best_price == float('inf'): daily_best_price = 0
        results['date'].append(date_str)
        results['price'].append(daily_best_price)
        results['airline'].append(daily_best_airline)
        results['stops'].append(daily_stops)
    return results

# --- GLAVNI APP ---
def main():
    st.title("✈️ AI Potovalni Agent")
    
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_params' not in st.session_state:
        st.session_state.search_params = {}

    data = load_data()
    if 'train' not in data or data['train'] is None:
        st.error("Učni podatki niso najdeni.")
        return
    
    df = data['train']
    model, encoders = train_model(df)
    
    st.sidebar.title("Navigacija")
    app_mode = st.sidebar.radio("Izberite način:", ["🔎 Iskanje Letov", "📊 Evalvacija Modela"])
    st.sidebar.markdown("---")

    # =========================================================================
    # STRAN 1: ISKANJE LETOV
    # =========================================================================
    if app_mode == "🔎 Iskanje Letov":
        st.sidebar.header("Nastavitve potovanja")
        trip_type = st.sidebar.radio("Tip potovanja:", ("Enosmerno", "Več mest (Multi-city)"))
        st.sidebar.markdown("---")
        max_stops = st.sidebar.slider("Največje število prestopov:", 0, 4, 1)
        
        if trip_type == "Enosmerno":
            st.header("🛫 Enosmerno potovanje")
            c1, c2, c3 = st.columns(3)
            source = c1.selectbox("Odhod", sorted(df['Source'].unique()))
            valid_destinations = sorted(df[df['Source'] == source]['Destination'].unique())
            
            if not valid_destinations:
                dest = c2.selectbox("Prihod", ["/"])
            else:
                dest = c2.selectbox("Prihod", valid_destinations)
            
            date = c3.date_input("Datum odhoda", datetime.date.today())
            
            if st.button("🔎 Poišči let"):
                st.markdown("---")
                if not valid_destinations or dest == "/":
                    st.error("Napaka pri destinaciji.")
                else:
                    offers = get_segment_offers(model, encoders, df, source, dest, date, max_stops)
                    st.session_state.search_results = offers
                    st.session_state.search_params = {
                        'source': source, 'dest': dest, 'date': date, 'max_stops': max_stops
                    }
                    st.session_state.search_performed = True

            if st.session_state.search_performed:
                offers = st.session_state.search_results
                params = st.session_state.search_params
                current_source = params['source']
                current_dest = params['dest']
                original_date = params['date']

                if not offers.empty:
                    offers = offers.sort_values('Cena')
                    best_price = offers.iloc[0]['Cena']
                    
                    st.success(f"Rezultati za: **{current_source} -> {current_dest}** na dan **{original_date.strftime('%d.%m.%Y')}**")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Najnižja cena", f"{best_price:.2f} €")
                    m2.metric("Št. ponudnikov", len(offers))
                    
                    st.subheader("Seznam ponudb")
                    display_df = offers[['Družba', 'Trajanje', 'Št. prestopov', 'Verjeten postanek', 'Cena']].copy()
                    st.dataframe(display_df.style.format({"Cena": "{:.2f} €"}), use_container_width=True)
                    
                    st.subheader("Primerjava cen")
                    fig = px.bar(
                        offers, x='Družba', y='Cena', color='Cena',
                        hover_data=['Št. prestopov', 'Trajanje', 'Verjeten postanek'],
                        title=f"Cene na dan {original_date.strftime('%d.%m.%Y')}",
                        text_auto='.2f', color_continuous_scale=CUSTOM_COLOR_SCALE
                    )
                    fig.update_layout(coloraxis_showscale=False) 
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📅 Koledar cen (Razišči druge datume)")
                    
                    c_cal1, c_cal2 = st.columns([1, 4])
                    with c_cal1:
                        cal_start_date = st.date_input("Začetek prikaza:", value=original_date, key="calendar_widget")
                    
                    with st.spinner("Posodabljam koledar..."):
                        days_to_show = 14
                        daily_data = get_best_daily_prices(
                            model, encoders, df, current_source, current_dest, 
                            cal_start_date, params['max_stops'], days=days_to_show
                        )
                        
                        if daily_data:
                            z_data = [daily_data['price']]
                            text_data = [[f"{p:.0f} €" for p in daily_data['price']]]
                            hover_text = [daily_data['airline']]
                            
                            fig_hm = go.Figure(data=go.Heatmap(
                                z=z_data, x=daily_data['date'], y=['Cena'],
                                text=text_data, texttemplate="%{text}", textfont={"size": 14},
                                colorscale=[[0.0, "#77DD77"], [0.5, "#FDFD96"], [1.0, "#FFB347"]],
                                showscale=False, customdata=hover_text,
                                hovertemplate="<b>Datum: %{x}</b><br>Družba: <b>%{customdata}</b><br>Cena: %{z:.2f} €<extra></extra>"
                            ))
                            fig_hm.update_layout(
                                height=150, xaxis_title="", yaxis_title="",
                                margin=dict(l=20, r=20, t=10, b=20), xaxis=dict(side="bottom")
                            )
                            st.plotly_chart(fig_hm, use_container_width=True)
                        else:
                            st.warning("Ni podatkov za izbrano obdobje.")
                else:
                    st.warning("Ni ponudb za izbrane kriterije.")

        elif trip_type == "Več mest (Multi-city)":
            st.header("🏙️ Potovanje med večimi kraji")
            st.info("Logika za Multi-city ostaja enaka.")

    # =========================================================================
    # STRAN 2: EVALVACIJA MODELA
    # =========================================================================
    elif app_mode == "📊 Evalvacija Modela":
        st.header("📊 Napredna Evalvacija Modela")
        st.markdown("Tukaj lahko eksperimentirate z delitvijo podatkov (Split), da vidite, kako se model obnaša.")
        
        # --- KONTROLE ZA SPLIT ---
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            # Drsnik za Test Size: 0.2, 0.25, 0.3, 0.35 ...
            test_size_input = st.slider(
                "Velikost testnega seta (Test Size)", 
                min_value=0.1, max_value=0.5, value=0.3, step=0.05,
                help="Privzeto 0.3 pomeni 30% podatkov za testiranje, 70% za učenje."
            )
        with col_ctrl2:
            # Random State (Seme za naključno mešanje)
            random_state_input = st.number_input(
                "Random State (Seme)", 
                value=42, step=1,
                help="Spremenite to številko, da premešate podatke na drugačen način."
            )

        if st.button("▶️ Zaženi evalvacijo s temi nastavitvami"):
            with st.spinner(f"Izvajam delitev {1-test_size_input:.2f} / {test_size_input:.2f} ..."):
                
                # 1. Priprava podatkov
                df_temp = df.copy()
                for col, enc in encoders.items():
                    if col in df_temp.columns:
                        df_temp[col] = enc.transform(df_temp[col].astype(str))
                
                X = df_temp.drop('Price', axis=1)
                if 'Route' in X.columns: X = X.drop('Route', axis=1)
                y = np.log1p(df_temp['Price']) 
                
                # 2. Uporaba uporabniških nastavitev (test_size in random_state)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size_input, 
                    random_state=random_state_input
                )
                
                # 3. Ponovno treniranje modela samo za evalvacijo (da je pošteno)
                # Uporabimo enak model kot v glavni aplikaciji
                model_eval = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
                model_eval.fit(X_train, y_train)
                
                # 4. Napoved
                y_pred_log = model_eval.predict(X_test)
                y_test_eur = np.expm1(y_test)
                y_pred_eur = np.expm1(y_pred_log)
                
                # 5. Metrike
                mae = mean_absolute_error(y_test_eur, y_pred_eur)
                mse = mean_squared_error(y_test_eur, y_pred_eur)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_eur, y_pred_eur)
                
                st.subheader(f"1. Rezultati (Split: {int((1-test_size_input)*100)}/{int(test_size_input*100)})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", f"{mae:.2f} €")
                c2.metric("RMSE", f"{rmse:.2f} €")
                c3.metric("R² Score", f"{r2:.4f}")
                c4.metric("MSE", f"{mse:.0f}")
                
                if r2 < 0.7:
                    st.error("R² je nizek. Morda je vzorec premajhen ali pa podatki preveč razpršeni.")
                elif r2 > 0.95:
                    st.warning("R² je sumljivo visok (> 95%). Preverite overfitting (prelegajanje).")
                else:
                    st.success("Rezultati izgledajo solidno.")

                st.markdown("---")
                
                # Grafi
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("**Dejanska vs. Napovedana cena**")
                    eval_df = pd.DataFrame({'Dejanska': y_test_eur, 'Napovedana': y_pred_eur})
                    fig_scatter = px.scatter(
                        eval_df, x='Dejanska', y='Napovedana', opacity=0.5, 
                        color_discrete_sequence=['#77DD77']
                    )
                    max_val = max(eval_df['Dejanska'].max(), eval_df['Napovedana'].max())
                    fig_scatter.add_shape(type="line", line=dict(dash='dash', color="gray"), x0=0, y0=0, x1=max_val, y1=max_val)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col_g2:
                    st.markdown("**Porazdelitev Napak**")
                    residuals = y_test_eur - y_pred_eur
                    fig_hist = px.histogram(
                        residuals, nbins=40,
                        color_discrete_sequence=['#FFB347']
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()