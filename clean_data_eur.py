import pandas as pd
import numpy as np

def clean_flight_data_eur(input_file, output_file, is_train=True):
    print(f"--- Obdelujem datoteko: {input_file} ---")
    
    # 1. Branje Excel datoteke
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"NAPAKA: Datoteke '{input_file}' ni mogoče najti.")
        return

    # 2. Odstranjevanje dvojnikov in praznih vrstic
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # 3. Pretvorba CENE v EVRE (Samo za učno množico, ker testna nima cene)
    if is_train and 'Price' in df.columns:
        print("  -> Pretvarjam cene v EUR (tečaj 0.011)...")
        # 1 INR je cca 0.011 EUR
        df['Price'] = df['Price'] * 0.011
        df['Price'] = df['Price'].round(2) # Zaokrožimo na 2 decimalki

    # 4. Urejanje DATUMOV (Dan, Mesec)
    # Dodamo tudi "Is_Weekend", ker to zelo vpliva na ceno
    df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
    df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
    
    # 0 = Ponedeljek, ... 5 = Sobota, 6 = Nedelja
    df['Weekday'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.weekday
    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    df.drop('Date_of_Journey', axis=1, inplace=True)
    df.drop('Weekday', axis=1, inplace=True) # Tega ne rabimo več, imamo Is_Weekend

    # 5. Urejanje ČASA (Ure, Minute)
    df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Dep_Min'] = pd.to_datetime(df['Dep_Time']).dt.minute
    df.drop('Dep_Time', axis=1, inplace=True)

    df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Arrival_Min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
    df.drop('Arrival_Time', axis=1, inplace=True)

    # 6. Trajanje leta v MINUTAH
    def convert_duration(duration):
        if 'h' not in duration:
            duration = '0h ' + duration
        if 'm' not in duration:
            duration = duration + ' 0m'
        
        parts = duration.split()
        h = int(parts[0].replace('h', ''))
        m = int(parts[1].replace('m', ''))
        return h * 60 + m

    df['Duration_Total_Mins'] = df['Duration'].apply(convert_duration)
    df.drop('Duration', axis=1, inplace=True)

    # 7. Postanki (Total_Stops)
    stops_map = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stops_map)

    # 8. Tekstovni popravki
    df['Destination'] = df['Destination'].replace('New Delhi', 'Delhi')
    df['Additional_Info'] = df['Additional_Info'].replace('No Info', 'No info')
    
    # Odstranimo Route, ker je podvojen podatek
    df.drop('Route', axis=1, inplace=True)

    # Shranjevanje
    print(f"  -> Shranjujem v: {output_file}")
    df.to_csv(output_file, index=False)
    print("  -> Končano.\n")

if __name__ == "__main__":
    # Očisti in pretvori UČNO množico
    clean_flight_data_eur('Data_Train.xlsx', 'Data_Train_Cleaned_EUR.csv', is_train=True)
    
    # Očisti TESTNO množico
    clean_flight_data_eur('Test_set.xlsx', 'Test_set_Cleaned_EUR.csv', is_train=False)