import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

def baseline_als(y, lam=10**5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def apply_preprocessing_pipeline(intensities):
    n_samples = intensities.shape[0]
    processed = np.zeros_like(intensities)
    spectra_clean = np.nan_to_num(intensities, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(n_samples):
        y = spectra_clean[i, :]
        if np.all(y == 0): continue
        
        baseline = baseline_als(y, lam=10**5, p=0.01)
        y_corrected = y - baseline
        
        processed[i, :] = savgol_filter(y_corrected, window_length=11, polyorder=2)

    mean_vals = processed.mean(axis=1, keepdims=True)
    std_vals = processed.std(axis=1, keepdims=True)
    std_vals[std_vals == 0] = 1
    processed_snv = (processed - mean_vals) / std_vals
    processed_snv = np.nan_to_num(processed_snv, nan=0.0)

    return processed_snv

def detect_spectrum_type(df):
    try:
        numeric_cols = []
        for c in df.columns:
            try:
                val = float(c)
                numeric_cols.append(val)
            except ValueError:
                pass
        
        if not numeric_cols:
            return None, []
        
        avg_wave = np.mean(numeric_cols)
        
        if avg_wave > 2200:
            return '2900', numeric_cols
        else:
            return '1500', numeric_cols
    except Exception:
        return None, []

def convert_txt_to_csv(filepath):
    print(f"Обработка сырого .txt файла: {filepath}")
    df_raw = pd.read_csv(filepath, sep=r'\s+', skiprows=1, names=['X', 'Y', 'Wave', 'Intensity'])
    
    unique_coords = df_raw[['X', 'Y']].drop_duplicates()
    n_pixels = len(unique_coords)
    n_total_rows = len(df_raw)
    n_waves = n_total_rows // n_pixels
    
    file_wave = df_raw['Wave'].values[:n_waves]
    
    if file_wave[0] > file_wave[-1]:
        file_wave = file_wave[::-1]
        intensity_matrix = df_raw['Intensity'].values.reshape(n_pixels, n_waves)[:, ::-1]
    else:
        intensity_matrix = df_raw['Intensity'].values.reshape(n_pixels, n_waves)
        
    df_wide = pd.DataFrame(intensity_matrix, columns=file_wave)
    
    df_wide.insert(0, 'X', unique_coords['X'].values)
    df_wide.insert(1, 'Y', unique_coords['Y'].values)
    
    new_filepath = filepath.rsplit('.', 1)[0] + '.csv'
    df_wide.to_csv(new_filepath, index=False)
    print(f"Файл успешно преобразован в матрицу и сохранен как: {new_filepath}")
    
    return new_filepath

def try_parse_float(val):
    try:
        return float(val)
    except:
        return None

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Введите путь к файлу (.txt или .csv): ").strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        print("Ошибка: Файл не найден.")
        return

    if file_path.lower().endswith('.txt'):
        try:
            file_path = convert_txt_to_csv(file_path)
        except Exception as e:
            print(f"Ошибка при преобразовании .txt файла: {e}")
            return

    try:
        df = pd.read_csv(file_path)
        print(f"Файл готов к анализу. Строк (спектров): {len(df)}")
    except Exception as e:
        print(f"Не удалось прочитать CSV: {e}")
        return

    spec_type, test_waves_float = detect_spectrum_type(df)
    
    if spec_type == '1500':
        print("Тип данных: Спектр 1500")
        model_file = 'model_1500.pkl'
        cols_file = 'cols_1500.pkl'
    elif spec_type == '2900':
        print("Тип данных: Спектр 2900")
        model_file = 'model_2900.pkl'
        cols_file = 'cols_2900.pkl'
    else:
        print("Ошибка: Не удалось определить тип спектра (нет числовых колонок).")
        return

    try:
        model = joblib.load(model_file)
        train_cols = joblib.load(cols_file) 
    except FileNotFoundError:
        print(f"Ошибка: Не найден файл {model_file} или {cols_file}. Убедитесь, что они лежат в той же папке.")
        return
    
    actual_cols = [c for c in df.columns if try_parse_float(c) in test_waves_float]
    test_intensities = df[actual_cols].values
    
    train_waves_float = np.array([float(c) for c in train_cols])
    
    print("Интерполяция спектров к стандартной размерности...")
    interpolator = interp1d(test_waves_float, test_intensities, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolated_intensities = interpolator(train_waves_float)

    print("Применение математических фильтров (ALS, SavGol, SNV)...")
    final_clean_matrix = apply_preprocessing_pipeline(interpolated_intensities)

    X_ready = pd.DataFrame(final_clean_matrix, index=df.index, columns=train_cols)
    
    try:
        preds = model.predict(X_ready)
        probs = model.predict_proba(X_ready)
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        preds = model.predict(X_ready.values)
        probs = model.predict_proba(X_ready.values)

    print("\n")

    for i, p in enumerate(preds[:10]):
        confidence = probs[i][p] * 100
        
        if p == 0: verdict = "Control"
        elif p == 1: verdict = "Endo"
        elif p == 2: verdict = "Exo"
        else: verdict = str(p)
            
        print(f"Пиксель {i+1:<5} | Прогноз: {verdict:<10} | Уверенность: {confidence:.2f}%")
        
    if len(preds) > 10:
        print(f"... и еще {len(preds) - 10} спектров.")

    unique_classes, counts = np.unique(preds, return_counts=True)
    majority_class_idx = unique_classes[np.argmax(counts)]
    
    classes_dict = {0: "Control", 1: "Endo", 2: "Exo"}
    print("\n" + "="*50)
    print(f"ИТОГОВЫЙ ДИАГНОЗ ДЛЯ ВСЕГО ОБРАЗЦА: {classes_dict.get(majority_class_idx, 'Unknown').upper()}")
    print("="*50 + "\n")

    res_df = pd.DataFrame({
        'Id': range(len(preds)),
        'Prediction_Class': preds,
        'Confidence': [probs[i][p] for i, p in enumerate(preds)]
    })
    res_df.to_csv('submission_result.csv', index=False)
    print("Подробный попиксельный результат сохранен в 'submission_result.csv'")

if __name__ == "__main__":
    main()