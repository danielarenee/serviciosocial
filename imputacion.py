"""
Script de Imputación y Distribución de Concentraciones (24 Compuestos ST)
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import eda

# Estilo gráfico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# IMPUTACIÓN US EPA (LPC / sqrt(2))

def imputar_dataset_epa(df_raw, df_detected, sample_ids):
    """
    Imputa concentraciones censuradas (< LPC) usando LPC / sqrt(2).
    """
    df_imputado = pd.DataFrame(index=df_raw.index)

    for i, row in df_raw.iterrows():
        lpc = row['lpc']
        is_cens = ~df_detected.loc[i, sample_ids].values
        raw_strs = row[sample_ids].astype(str).str.replace('<', '').str.replace(' ', '')
        num_vals = pd.to_numeric(raw_strs, errors='coerce').values
        imputed_vals = np.where(is_cens, lpc / np.sqrt(2.0), num_vals)

        for col_idx, col in enumerate(sample_ids):
            df_imputado.loc[i, col] = imputed_vals[col_idx]

    return df_imputado


# PANEL DE HISTOGRAMAS (JUSTIFICACIÓN HORNUNG & REED)

def graficar_histogramas_imputados(df_raw, df_detected, sample_ids, output_dir='figuras'):
    """
    Histogramas de los 10 compuestos que requieren imputación (< LPC).
    """
    os.makedirs(output_dir, exist_ok=True)

    cens_indices = [i for i, row in df_raw.iterrows() if (~df_detected.loc[i, sample_ids]).sum() > 0]
    df_cens = df_raw.loc[cens_indices].sort_values(by=['familia', 'id_compuesto'])

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    axes = axes.flatten()

    for idx, (i, row) in enumerate(df_cens.iterrows()):
        nom = row['nombre_compuesto']
        fam = 'PFAS' if row['familia'] == 'PFAS' else 'Restr'
        color_base = '#1f77b4' if fam == 'PFAS' else '#e6550d'
        color_line = '#08519c' if fam == 'PFAS' else '#a63603'

        raw_strs = row[sample_ids].astype(str).str.replace('<', '').str.replace(' ', '')
        vals = pd.to_numeric(raw_strs, errors='coerce').dropna().values
        is_cens = (~df_detected.loc[i, sample_ids]).values.flatten()
        det_vals = vals[~is_cens]

        gsd = np.exp(np.std(np.log(det_vals), ddof=1)) if len(det_vals) > 1 else 1.0

        ax = axes[idx]
        sns.histplot(
            det_vals,
            bins=4,
            kde=True,
            color=color_base,
            edgecolor='white',
            alpha=0.55,
            ax=ax,
            line_kws={'linewidth': 1.8, 'color': color_line}
        )
        sns.rugplot(det_vals, color='#b30000', height=0.15, linewidth=2.5, ax=ax, label=f'Detectados ({len(det_vals)}/7)')

        ax.set_title(f'[{fam}] {nom[:22]}\n(GSD = {gsd:.2f})',
                     fontsize=10.5, fontweight='bold')
        ax.set_xlabel(f'Concentración ({row.unidad})', fontsize=10)
        ax.set_ylabel('Frecuencia / Densidad', fontsize=10)
        ax.tick_params(labelsize=9.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        lpc_val = row['lpc']
        ax.axvline(lpc_val, color='#d95f02', linestyle='--', linewidth=1.2, label=f'LPC = {lpc_val}')
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout(pad=2.5)

    ruta = os.path.join(output_dir, '05_panel_histogramas_compuestos_imputados.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta}")


# BOXPLOTS DE CONCENTRACIÓN (ESCALA LOGARÍTMICA, ORDENADOS POR MEDIANA)

def graficar_boxplots(df_imputado, df_raw, sample_ids, output_dir='figuras'):
    """
    Genera boxplots en escala logarítmica ordenados por mediana con puntos individuales (strip plot) y línea de LPC.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. BOXPLOTS PFAS (11 compuestos, ng/L)
    pfas_indices = df_raw[df_raw['familia'] == 'PFAS'].index
    df_pfas = df_imputado.loc[pfas_indices, sample_ids].copy()
    df_pfas['nombre'] = df_raw.loc[pfas_indices, 'nombre_compuesto']

    orden_pfas = df_pfas.set_index('nombre')[sample_ids].median(axis=1).sort_values(ascending=False).index
    df_pfas_melt = pd.melt(df_pfas, id_vars=['nombre'], value_vars=sample_ids, var_name='muestra', value_name='concentracion')

    plt.figure(figsize=(13, 9))
    ax = sns.boxplot(
        data=df_pfas_melt,
        y='nombre',
        x='concentracion',
        order=orden_pfas,
        hue='nombre',
        legend=False,
        palette='Blues_r',
        fliersize=0,
        linewidth=1.0,
        boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=df_pfas_melt,
        y='nombre',
        x='concentracion',
        order=orden_pfas,
        color='#08306b',
        alpha=0.65,
        size=6,
        jitter=0.2
    )

    ax.axvline(2.00, color='#d95f02', linestyle='--', linewidth=1.2, label='LPC = 2.00 ng/L')
    ax.set_xscale('log')
    ax.set_xlabel('Concentración (ng/L) — Escala Logarítmica', fontsize=13, fontweight='bold')
    ax.set_ylabel('Compuesto PFAS', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout(pad=2.0)
    ruta_pfas = os.path.join(output_dir, '06_boxplots_concentraciones_pfas.png')
    plt.savefig(ruta_pfas, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_pfas}")

    # 2. BOXPLOTS SUSTANCIAS RESTRINGIDAS (13 compuestos, µg/mL)
    restr_indices = df_raw[df_raw['familia'] == 'Sustancias Restringidas'].index
    df_restr = df_imputado.loc[restr_indices, sample_ids].copy()
    df_restr['nombre'] = df_raw.loc[restr_indices, 'nombre_compuesto']

    orden_restr = df_restr.set_index('nombre')[sample_ids].median(axis=1).sort_values(ascending=False).index
    df_restr_melt = pd.melt(df_restr, id_vars=['nombre'], value_vars=sample_ids, var_name='muestra', value_name='concentracion')

    plt.figure(figsize=(13, 10))
    ax = sns.boxplot(
        data=df_restr_melt,
        y='nombre',
        x='concentracion',
        order=orden_restr,
        hue='nombre',
        legend=False,
        palette='Oranges_r',
        fliersize=0,
        linewidth=1.0,
        boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=df_restr_melt,
        y='nombre',
        x='concentracion',
        order=orden_restr,
        color='#7f2704',
        alpha=0.65,
        size=6,
        jitter=0.2
    )

    ax.axvline(0.0100, color='#d95f02', linestyle='--', linewidth=1.2, label='LPC = 0.0100 µg/mL')
    ax.set_xscale('log')
    ax.set_xlabel('Concentración (µg/mL) — Escala Logarítmica', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sustancia Restringida', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout(pad=2.0)
    ruta_restr = os.path.join(output_dir, '07_boxplots_concentraciones_restringidas.png')
    plt.savefig(ruta_restr, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {ruta_restr}")


# EXPORTACIÓN DEL DATASET IMPUTADO

def exportar_dataset_imputado(df_raw, sample_ids, df_imputado, ruta_csv_base='muestras_24_st.csv', output_csv='muestras_24_st_imputado_epa.csv', output_dir='outputs'):
    """
    Exporta dataset con concentraciones imputadas conservando encabezados originales.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(ruta_csv_base, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    filas_exportar = [reader[0], reader[1]]
    for i, row in df_raw.iterrows():
        id_val = str(int(row['id_compuesto'])) if pd.notnull(row['id_compuesto']) else ''
        nombre_val = row['nombre_compuesto']
        vals = [f"{df_imputado.loc[i, s]:.4f}" for s in sample_ids]
        filas_exportar.append([id_val, nombre_val] + vals)

    rutas = [output_csv, os.path.join(output_dir, output_csv)]
    for ruta in set(rutas):
        with open(ruta, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(filas_exportar)
        print(f"-> Dataset imputado exportado: {ruta}")


# MAIN

def main():
    # 1. Cargar datos
    df_raw, df_muestras_info, df_detected, df_concentraciones, sample_ids = eda.cargar_datos('muestras_24_st.csv')

    # 2. Imputar dataset vía US EPA (LPC / sqrt(2))
    df_imputado_epa = imputar_dataset_epa(df_raw, df_detected, sample_ids)

    # 3. Exportar dataset imputado
    print("EXPORTANDO DATASET IMPUTADO...")
    exportar_dataset_imputado(df_raw, sample_ids, df_imputado_epa, ruta_csv_base='muestras_24_st.csv', output_csv='muestras_24_st_imputado_epa.csv', output_dir='outputs')

    # 4. Generar visualizaciones
    print("GENERANDO VISUALIZACIONES...")
    graficar_histogramas_imputados(df_raw, df_detected, sample_ids, output_dir='figuras')
    graficar_boxplots(df_imputado_epa, df_raw, sample_ids, output_dir='figuras')

    print("[ÉXITO] Imputación y boxplots completados.")


if __name__ == '__main__':
    main()
