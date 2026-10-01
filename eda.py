"""
Script de EDA
tenemos 7 sin tratar, 5 tratadas y 1 lodo 
"""
#%%

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# CARGA DE DATOS

def cargar_datos(ruta_csv='muestras.csv'):
    """
    separa metadatos de compuestos, códigos de trazabilidad y genera
    matrices de concentraciones numéricas y estados de detección
    """
    # leer archivo crudo para separar encabezados
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        lineas = [line.strip() for line in f.readlines() if line.strip()]

    # fila 1: IDs de muestra
    header_muestras = [col.strip() for col in lineas[0].split(',')]
    # fila 2: códigos CIATEC
    header_ciatec = [col.strip() for col in lineas[1].split(',')]

    # identificar columnas de muestras
    sample_ids = header_muestras[2:]
    ciatec_map = dict(zip(sample_ids, header_ciatec[2:]))

    # cargar filas de compuestos
    df_raw = pd.read_csv(
        ruta_csv,
        skiprows=2,
        header=None,
        names=['id_compuesto', 'nombre_compuesto'] + sample_ids,
        encoding='utf-8'
    )

    # limpieza básica de identificadores y nombres
    df_raw['id_compuesto'] = pd.to_numeric(df_raw['id_compuesto'], errors='coerce')
    df_raw['nombre_compuesto'] = df_raw['nombre_compuesto'].astype(str).str.strip()

    # asignación de familia analítica, unidades y LPC
    # IDs 1-29: PFAS (ng/L, LPC = 2.00)
    # IDs 30-97: Sustancias restringidas (µg/mL, LPC = 0.0100)
    df_raw['familia'] = np.where(df_raw['id_compuesto'] <= 29, 'PFAS', 'Sustancias Restringidas')
    df_raw['unidad'] = np.where(df_raw['id_compuesto'] <= 29, 'ng/L', 'µg/mL')
    df_raw['lpc'] = np.where(df_raw['id_compuesto'] <= 29, 2.00, 0.0100)

    # metadatos de muestras estructurados
    df_muestras_info = pd.DataFrame({'muestra_id': sample_ids})
    df_muestras_info['codigo_ciatec'] = df_muestras_info['muestra_id'].map(ciatec_map)
    df_muestras_info['lavanderia'] = df_muestras_info['muestra_id'].apply(lambda x: re.match(r'(L\d+)', x).group(1))
    
    def extraer_tipo(x):
        if '_ST' in x:
            return 'ST'  # Sin tratar
        elif '_T' in x:
            return 'T'   # Tratada
        elif '_L' in x:
            return 'L'   # Lodos
        return 'Otro'
    
    df_muestras_info['tipo'] = df_muestras_info['muestra_id'].apply(extraer_tipo)
    df_muestras_info['tipo_desc'] = df_muestras_info['tipo'].map({
        'ST': 'Agua Residual Sin Tratar',
        'T': 'Agua Tratada',
        'L': 'Lodos'
    })
    df_muestras_info['replica'] = df_muestras_info['muestra_id'].apply(
        lambda x: re.search(r'_(\d+)$', x).group(1) if re.search(r'_(\d+)$', x) else None
    )

    # Matriz de detección booleana (True = Detectado, False = Censurado)

    """
    me hubiera gustado usar regresión sobre estadísticos de orden o algún otro método pero la tasa de 
    censura debe ser <50% y n>= 8-10 por grupo , pero tenemos aprox 60% censura y n=7 en sin tratar

    ningún métdo de imputación va a ser confiable
    """

    df_detected = pd.DataFrame(index=df_raw.index)
    df_concentraciones = pd.DataFrame(index=df_raw.index)

    for col in sample_ids:
        # Limpiar texto del valor
        raw_vals = df_raw[col].astype(str).str.strip().str.replace(' ', '')
        is_censored = raw_vals.str.contains('<')
        df_detected[col] = ~is_censored

        # Convertir a flotante
        # Para censurados, tomamos LPC/2
        numeric_vals = pd.to_numeric(raw_vals.str.replace('<', ''), errors='coerce')
        imputed_vals = np.where(is_censored, df_raw['lpc'] / 2.0, numeric_vals)
        df_concentraciones[col] = imputed_vals

    return df_raw, df_muestras_info, df_detected, df_concentraciones, sample_ids


# 2. FRECUENCIA DE DETECCIÓN

def calcular_resumen_deteccion(df_raw, df_detected, df_muestras_info, sample_ids):
    """
    por tipo de matriz (Aguas Crudas ST, Tratadas T, Lodos L)
    """
    columnas_st = df_muestras_info[df_muestras_info['tipo'] == 'ST']['muestra_id'].tolist()
    columnas_t = df_muestras_info[df_muestras_info['tipo'] == 'T']['muestra_id'].tolist()
    columnas_l = df_muestras_info[df_muestras_info['tipo'] == 'L']['muestra_id'].tolist()

    resumen = pd.DataFrame({
        'id_compuesto': df_raw['id_compuesto'],
        'nombre_compuesto': df_raw['nombre_compuesto'],
        'familia': df_raw['familia'],
        'unidad': df_raw['unidad'],
        'lpc': df_raw['lpc'],
        'det_ST': df_detected[columnas_st].sum(axis=1),
        'total_muestras_ST': len(columnas_st),
        'pct_det_ST': (df_detected[columnas_st].mean(axis=1) * 100).round(1),
        'det_total': df_detected[sample_ids].sum(axis=1),
        'pct_det_total': (df_detected[sample_ids].mean(axis=1) * 100).round(1),
        'det_T': df_detected[columnas_t].sum(axis=1),
        'det_L': df_detected[columnas_l].sum(axis=1) if columnas_l else 0,
    })

    return resumen


# 3. GENERACIÓN DE VISUALIZACIONES

def graficar_tasa_deteccion(resumen, output_dir='figuras'):
    """
    gráfico de barras con los contaminantes más frecuentes
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar solo compuestos que se detectaron en agua cruda (ST)
    detectados = resumen[resumen['det_ST'] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)
    colores_familia = {'PFAS': '#1f77b4', 'Sustancias Restringidas': '#e6550d'}

    for idx, (familia, ax) in enumerate(zip(['PFAS', 'Sustancias Restringidas'], axes)):
        sub_df = detectados[detectados['familia'] == familia].sort_values(by='pct_det_ST', ascending=True)
        total_familia = len(resumen[resumen['familia'] == familia])

        if len(sub_df) == 0:
            ax.text(0.5, 0.5, 'No se detectaron compuestos en agua cruda', ha='center', va='center')
            continue

        y_positions = np.arange(len(sub_df))
        bars = ax.barh(y_positions, sub_df['pct_det_ST'], color=colores_familia[familia], alpha=0.85, height=0.65)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(sub_df['nombre_compuesto'], fontsize=11)
        ax.set_xlabel('% Muestras de Agua Cruda con Detección (> LPC)', fontsize=12, fontweight='bold')
        ax.set_title(f'Familia: {familia}\n({len(sub_df)} detectados en ST de {total_familia} monitoreados)',
                     fontsize=13, fontweight='bold', pad=12)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='x', labelsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 2.0, bar.get_y() + bar.get_height() / 2, f'{width:.1f}% ({int(round(width*7/100))}/7)',
                    ha='left', va='center', fontsize=10, color='#222222', fontweight='medium')

    plt.tight_layout(pad=2.5)
    
    ruta_guardado = os.path.join(output_dir, '01_tasa_deteccion_agua_cruda.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def graficar_huella_quimica_heatmap(df_raw, df_detected, df_muestras_info, output_dir='figuras'):
    """
    mapa de calor binario (Detectado vs < LPC) ordenado por lavandería y tipo
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar columnas de Agua Cruda (ST)
    muestras_st = df_muestras_info[df_muestras_info['tipo'] == 'ST'].sort_values(by=['lavanderia', 'replica'])
    cols_st = muestras_st['muestra_id'].tolist()

    # Identificar compuestos detectados en al menos 1 muestra ST
    mask_detectados_st = df_detected[cols_st].sum(axis=1) > 0
    num_detectados = mask_detectados_st.sum()
    num_censurados_100 = (~mask_detectados_st).sum()

    df_st_filtrado = df_detected.loc[mask_detectados_st, cols_st].copy()

    # Nombres de fila con familia
    nombres_compuestos = (
        "[" + df_raw.loc[mask_detectados_st, 'familia'].apply(lambda x: 'PFAS' if x == 'PFAS' else 'Restr') + "] " +
        df_raw.loc[mask_detectados_st, 'nombre_compuesto']
    )
    df_st_filtrado.index = nombres_compuestos

    # Ordenar por número de detecciones descendente
    orden_compuestos = df_st_filtrado.sum(axis=1).sort_values(ascending=False).index
    df_st_filtrado = df_st_filtrado.loc[orden_compuestos]

    plt.figure(figsize=(12, 11))
    
    # 0 = No Detectado (< LPC), 1 = Detectado (> LPC)
    cmap = sns.color_palette(['#eceff1', '#0b5394'])
    
    ax = sns.heatmap(
        df_st_filtrado.astype(int),
        cmap=cmap,
        cbar_kws={'label': 'Condición Analítica', 'ticks': [0.25, 0.75], 'shrink': 0.5},
        linewidths=0.6,
        linecolor='#cfd8dc'
    )
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(['< LPC (Censurado)', '> LPC (Detectado)'], fontsize=11)
    cbar.ax.set_ylabel('Condición Analítica', fontsize=11)

    plt.xlabel('Muestra de Agua Cruda (ST)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Compuesto Detectado', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=10)
    
    plt.tight_layout(pad=2.0)
    ruta_guardado = os.path.join(output_dir, '02_huella_quimica_agua_cruda.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def graficar_huella_quimica_completa(df_raw, df_detected, df_muestras_info, output_dir='figuras'):
    """
    mapa de calor binario completo con los 97 compuestos monitoreados en agua cruda (ST)
    permite visualizar la proporción real de datos censurados (< LPC, 76%) vs detectados
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar columnas de Agua Cruda (ST)
    muestras_st = df_muestras_info[df_muestras_info['tipo'] == 'ST'].sort_values(by=['lavanderia', 'replica'])
    cols_st = muestras_st['muestra_id'].tolist()

    df_st_completo = df_detected[cols_st].copy()

    # Nombres de fila con ID y familia
    nombres_compuestos = (
        "[" + df_raw['familia'].apply(lambda x: 'PFAS' if x == 'PFAS' else 'Restr') + " " +
        df_raw['id_compuesto'].astype(str) + "] " + df_raw['nombre_compuesto']
    )
    df_st_completo.index = nombres_compuestos

    # Ordenar por número de detecciones descendente para agrupar detectados arriba y censurados abajo
    orden_compuestos = df_st_completo.sum(axis=1).sort_values(ascending=False).index
    df_st_completo = df_st_completo.loc[orden_compuestos]

    plt.figure(figsize=(13, 24))
    
    # 0 = No Detectado (< LPC), 1 = Detectado (> LPC)
    cmap = sns.color_palette(['#eceff1', '#0b5394'])
    
    ax = sns.heatmap(
        df_st_completo.astype(int),
        cmap=cmap,
        cbar_kws={'label': 'Condición Analítica', 'ticks': [0.25, 0.75], 'shrink': 0.3},
        linewidths=0.4,
        linecolor='#cfd8dc'
    )
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(['< LPC (Censurado)', '> LPC (Detectado)'], fontsize=11)
    cbar.ax.set_ylabel('Condición Analítica', fontsize=11)

    # Línea divisoria horizontal entre los 34 detectados y los 63 no detectados
    ax.axhline(34, color='#d32f2f', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(cols_st) / 2, 34.5, '34 Compuestos Detectados (>=1 muestra ST)  |  63 Compuestos 100% Censurados (<LPC)',
            color='#b71c1c', fontsize=10, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#ef5350', alpha=0.9))

    plt.xlabel('Muestra de Agua Cruda (ST)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Compuesto Monitoreado (97 en total)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=8)
    
    plt.tight_layout(pad=2.0)
    ruta_guardado = os.path.join(output_dir, '02b_huella_quimica_completa_97_compuestos.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def _preparar_consistencia(df_raw, df_detected, df_muestras_info):
    """Calcula la presencia de cada compuesto por lavandería (ST). Uso interno."""
    muestras_st = df_muestras_info[df_muestras_info['tipo'] == 'ST']
    lavanderias = sorted(muestras_st['lavanderia'].unique())

    deteccion_por_lav = pd.DataFrame(index=df_raw.index)
    for lav in lavanderias:
        cols_lav = muestras_st[muestras_st['lavanderia'] == lav]['muestra_id'].tolist()
        deteccion_por_lav[lav] = df_detected[cols_lav].any(axis=1)

    num_lavs_presente = deteccion_por_lav.sum(axis=1)
    df_consistencia = pd.DataFrame({
        'id_compuesto': df_raw['id_compuesto'],
        'nombre_compuesto': df_raw['nombre_compuesto'],
        'familia': df_raw['familia'],
        'num_lavanderias': num_lavs_presente
    })
    return df_consistencia[df_consistencia['num_lavanderias'] > 0].sort_values(
        by=['num_lavanderias', 'familia'], ascending=[False, True]
    )


def graficar_consistencia_pie(df_raw, df_detected, df_muestras_info, output_dir='figuras'):
    """
    Gráfico de pastel: distribución de los 34 contaminantes según cuántas lavanderías los tienen.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_consistencia = _preparar_consistencia(df_raw, df_detected, df_muestras_info)

    conteo_dist = df_consistencia['num_lavanderias'].value_counts().sort_index(ascending=False)
    colores_pie = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
    etiquetas = [f'En las {n} lavanderías' if n == 5 else f'En {n} lavanderías' for n in conteo_dist.index]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(
        conteo_dist,
        labels=etiquetas,
        autopct='%1.1f%%',
        startangle=140,
        colors=colores_pie[:len(conteo_dist)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 13}
    )
    for text in ax.texts:
        text.set_fontsize(13)

    plt.tight_layout(pad=2.0)
    ruta_guardado = os.path.join(output_dir, '03a_consistencia_pie.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def graficar_consistencia_barras(df_raw, df_detected, df_muestras_info, output_dir='figuras'):
    """
    Gráfico de barras horizontales: ubicuidad de cada compuesto en las 5 lavanderías.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_consistencia = _preparar_consistencia(df_raw, df_detected, df_muestras_info)

    top_compuestos = df_consistencia.sort_values(by='num_lavanderias', ascending=True)
    y_pos = np.arange(len(top_compuestos))
    bar_colors = ['#08519c' if x == 5 else ('#3182bd' if x == 4 else '#74a9cf') for x in top_compuestos['num_lavanderias']]

    fig, ax = plt.subplots(figsize=(13, 12))
    ax.barh(y_pos, top_compuestos['num_lavanderias'], color=bar_colors, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_compuestos['nombre_compuesto'], fontsize=11)
    ax.set_xlabel('Número de Lavanderías con Presencia (máximo 5)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 5.8)
    ax.set_xticks(range(1, 6))
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for i, val in enumerate(top_compuestos['num_lavanderias']):
        fam = top_compuestos.iloc[i]['familia']
        fam_tag = 'PFAS' if fam == 'PFAS' else 'Restr'
        ax.text(val + 0.08, i, f'{val}/5 ({fam_tag})', va='center', fontsize=10, color='#333333')

    plt.tight_layout(pad=2.5)
    ruta_guardado = os.path.join(output_dir, '03b_consistencia_barras.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def graficar_huella_24_compuestos(df_raw, df_detected, df_muestras_info, mask_24, output_dir='figuras'):
    """
    mapa de calor para el universo filtrado de 24 compuestos prioritarios (>= 3 lavanderías)
    """
    os.makedirs(output_dir, exist_ok=True)

    muestras_st = df_muestras_info[df_muestras_info['tipo'] == 'ST'].sort_values(by=['lavanderia', 'replica'])
    cols_st = muestras_st['muestra_id'].tolist()

    df_24 = df_detected.loc[mask_24, cols_st].copy()

    nombres_compuestos = (
        "[" + df_raw.loc[mask_24, 'familia'].apply(lambda x: 'PFAS' if x == 'PFAS' else 'Restr') + " " +
        df_raw.loc[mask_24, 'id_compuesto'].astype(str) + "] " + df_raw.loc[mask_24, 'nombre_compuesto']
    )
    df_24.index = nombres_compuestos

    orden = df_24.sum(axis=1).sort_values(ascending=False).index
    df_24 = df_24.loc[orden]

    plt.figure(figsize=(11, 10))
    cmap = sns.color_palette(['#eceff1', '#08519c'])

    ax = sns.heatmap(
        df_24.astype(int),
        cmap=cmap,
        cbar_kws={'label': 'Condición Analítica', 'ticks': [0.25, 0.75], 'shrink': 0.5},
        linewidths=0.6,
        linecolor='#cfd8dc'
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(['< LPC (Censurado)', '> LPC (Detectado)'], fontsize=11)
    cbar.ax.set_ylabel('Condición Analítica', fontsize=11)

    plt.xlabel('Muestra de Agua Cruda (ST)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Compuesto Prioritario', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=10)

    plt.tight_layout(pad=2.0)
    ruta_guardado = os.path.join(output_dir, '04_huella_quimica_24_compuestos.png')
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_guardado}")


def exportar_muestras_filtradas(df_raw, df_detected, df_muestras_info, ruta_csv_origen='muestras.csv', min_lavanderias=3, output_csv='muestras_24.csv', output_csv_st='muestras_24_st.csv', output_dir='outputs'):
    """
    Filtra los compuestos que están presentes en al menos `min_lavanderias` (>=3 de 5 lavanderías)
    y exporta:
      1. `muestras_24.csv`: universo de 24 compuestos con todas las muestras (ST, T, L).
      2. `muestras_24_st.csv`: universo de 24 compuestos EXCLUSIVAMENTE con muestras sin tratar (ST).
    Ambos conservan los encabezados de 2 niveles y el formato del archivo original.
    """
    import csv
    os.makedirs(output_dir, exist_ok=True)

    # Identificar lavanderías con presencia en ST
    muestras_st_df = df_muestras_info[df_muestras_info['tipo'] == 'ST']
    muestras_st_ids = muestras_st_df['muestra_id'].tolist()
    lavanderias = sorted(muestras_st_df['lavanderia'].unique())

    deteccion_por_lav = pd.DataFrame(index=df_raw.index)
    for lav in lavanderias:
        cols_lav = muestras_st_df[muestras_st_df['lavanderia'] == lav]['muestra_id'].tolist()
        deteccion_por_lav[lav] = df_detected[cols_lav].any(axis=1)

    num_lavs_presente = deteccion_por_lav.sum(axis=1)
    mask_filtrada = num_lavs_presente >= min_lavanderias
    num_seleccionados = mask_filtrada.sum()

    # Leer el archivo original con csv.reader para manejar comillas y caracteres especiales
    with open(ruta_csv_origen, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    h1 = reader[0]
    h2 = reader[1]
    data_rows = reader[2:]

    # Índices de columnas ST (0: id, 1: nombre, y columnas correspondientes a ST)
    st_indices = [0, 1] + [i for i, col in enumerate(h1) if col.strip() in muestras_st_ids]

    # Filas para el archivo completo de 24 compuestos
    rows_24_completo = [h1, h2]
    # Filas para el archivo exclusivo ST de 24 compuestos
    rows_24_st = [[h1[i] for i in st_indices], [h2[i] for i in st_indices]]

    for idx, row in enumerate(data_rows):
        if mask_filtrada.iloc[idx]:
            rows_24_completo.append(row)
            rows_24_st.append([row[i] for i in st_indices])

    # 1. Exportar muestras_24.csv (todas las muestras)
    for ruta in set([output_csv, os.path.join(output_dir, output_csv)]):
        with open(ruta, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows_24_completo)
        print(f"-> Archivo CSV filtrado exportado: {ruta} ({num_seleccionados} compuestos, todas las muestras)")

    # 2. Exportar muestras_24_st.csv (solo muestras ST)
    for ruta in set([output_csv_st, os.path.join(output_dir, output_csv_st)]):
        with open(ruta, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows_24_st)
        print(f"-> Archivo CSV ST exportado: {ruta} ({num_seleccionados} compuestos, 7 muestras ST)")

    # 3. Exportar resumen tabular estructurado en outputs/
    df_resumen_24 = df_raw.loc[mask_filtrada, ['id_compuesto', 'nombre_compuesto', 'familia', 'unidad', 'lpc']].copy()
    df_resumen_24['num_lavanderias_st'] = num_lavs_presente[mask_filtrada]
    df_resumen_24 = df_resumen_24.sort_values(by=['num_lavanderias_st', 'familia'], ascending=[False, True])
    df_resumen_24.to_csv(os.path.join(output_dir, 'resumen_24_compuestos.csv'), index=False, encoding='utf-8')
    print(f"-> Resumen estructurado exportado: {os.path.join(output_dir, 'resumen_24_compuestos.csv')}")

    return mask_filtrada, df_resumen_24

#%%

# MAIN

def main():
    print(" ==== ANÁLISIS EXPLORATORIO DE DATOS =====")

    # 1. Cargar datos
    df_raw, df_muestras_info, df_detected, df_concentraciones, sample_ids = cargar_datos('muestras.csv')
    
    muestras_st = df_muestras_info[df_muestras_info['tipo'] == 'ST']['muestra_id'].tolist()
    total_celdas_st = len(df_raw) * len(muestras_st)
    censuradas_st = (~df_detected[muestras_st]).values.sum()
    detectadas_st_celdas = df_detected[muestras_st].values.sum()

    print(f"\n[OK] Datos cargados correctamente:")
    print(f"     - Total de compuestos analizados: {len(df_raw)} (IDs 1-29 PFAS, IDs 30-97 Sustancias Restringidas)")
    print(f"     - Muestras de Agua Cruda (ST): {len(muestras_st)} muestras de 5 lavanderías ({', '.join(muestras_st)})")
    print(f"     - Diagnóstico de Censura en ST: {censuradas_st}/{total_celdas_st} datos son < LPC ({censuradas_st/total_celdas_st*100:.1f}% censura)")
    print(f"     - Datos cuantificados en ST: {detectadas_st_celdas}/{total_celdas_st} ({detectadas_st_celdas/total_celdas_st*100:.1f}%)")

    # 2. Resumen de detección
    resumen = calcular_resumen_deteccion(df_raw, df_detected, df_muestras_info, sample_ids)
    
    detectados_st = resumen[resumen['det_ST'] > 0]
    no_detectados_st = resumen[resumen['det_ST'] == 0]

    print("\n" + "=" * 75)
    print(" HALLAZGOS PRINCIPALES DE DETECCIÓN EN AGUA CRUDA (ST):")
    print("=" * 75)
    print(f"• De 97 compuestos monitoreados:")
    print(f"  - {len(detectados_st)} compuestos ({len(detectados_st)/97*100:.1f}%) se detectan en AL MENOS 1 muestra de agua cruda.")
    print(f"  - {len(no_detectados_st)} compuestos ({len(no_detectados_st)/97*100:.1f}%) son 100% CENSURADOS (<LPC) en agua cruda.")

    print("\n• Desglose por Familia Analítica en Agua Cruda (ST):")
    for fam in ['PFAS', 'Sustancias Restringidas']:
        sub = resumen[resumen['familia'] == fam]
        det = sub[sub['det_ST'] > 0]
        print(f"  - {fam}: {len(det)} de {len(sub)} detectados ({len(det)/len(sub)*100:.1f}%) | {len(sub)-len(det)} nunca detectados")

    print("\n• Top Contaminantes más Frecuentes en Agua Cruda (ST):")
    cols_mostrar = ['id_compuesto', 'nombre_compuesto', 'familia', 'det_ST', 'pct_det_ST']
    print(detectados_st.sort_values(by='pct_det_ST', ascending=False)[cols_mostrar].head(15).to_string(index=False))

    # 3. Exportación de universo reducido (>= 3 lavanderías -> 24 compuestos)
    print("\n" + "=" * 75)
    print(" REDUCCIÓN DE UNIVERSO Y EXPORTACIÓN (>= 3 LAVANDERÍAS)...")
    print("=" * 75)
    mask_24, df_resumen_24 = exportar_muestras_filtradas(
        df_raw, df_detected, df_muestras_info,
        ruta_csv_origen='muestras.csv',
        min_lavanderias=3,
        output_csv='muestras_24.csv',
        output_dir='outputs'
    )
    
    # Exportar también resumen completo de detección
    resumen.to_csv('outputs/resumen_deteccion_completo.csv', index=False, encoding='utf-8')
    print("-> Resumen completo exportado: outputs/resumen_deteccion_completo.csv")

    pfas_24 = len(df_resumen_24[df_resumen_24['familia'] == 'PFAS'])
    restr_24 = len(df_resumen_24[df_resumen_24['familia'] == 'Sustancias Restringidas'])
    print(f"\n[OK] Universo reducido a {len(df_resumen_24)} compuestos prioritarios:")
    print(f"     - PFAS ({pfas_24}): {', '.join(df_resumen_24[df_resumen_24['familia'] == 'PFAS']['nombre_compuesto'].tolist()[:4])}...")
    print(f"     - Sustancias Restringidas ({restr_24}): {', '.join(df_resumen_24[df_resumen_24['familia'] == 'Sustancias Restringidas']['nombre_compuesto'].tolist()[:4])}...")

    # 4. Generar visualizaciones
    print("\n" + "=" * 75)
    print(" GENERANDO VISUALIZACIONES...")
    print("=" * 75)
    graficar_tasa_deteccion(resumen, output_dir='figuras')
    graficar_huella_quimica_heatmap(df_raw, df_detected, df_muestras_info, output_dir='figuras')
    graficar_huella_quimica_completa(df_raw, df_detected, df_muestras_info, output_dir='figuras')
    graficar_consistencia_pie(df_raw, df_detected, df_muestras_info, output_dir='figuras')
    graficar_consistencia_barras(df_raw, df_detected, df_muestras_info, output_dir='figuras')
    graficar_huella_24_compuestos(df_raw, df_detected, df_muestras_info, mask_24, output_dir='figuras')

    print("\n[ÉXITO] EDA de agua cruda completado. Revisa las figuras en './figuras' y outputs en './outputs'.")


if __name__ == '__main__':
    main()

# %%
