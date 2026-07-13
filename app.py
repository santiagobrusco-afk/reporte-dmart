import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="App de Extensiones Dmart", layout="wide")

st.title("📊 Análisis de Performance: Extensiones Horarias")
st.markdown("Herramienta interactiva para evaluar el impacto de las extensiones de madrugada.")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("1. Carga de Datos")
ops_file = st.sidebar.file_uploader("Subir CSV de Operaciones", type="csv")
audit_file = st.sidebar.file_uploader("Subir CSV de Auditoría", type="csv")

st.sidebar.header("2. Fechas de Análisis")
fecha_corte_input = st.sidebar.date_input("Fecha de Corte (Inicio del AFTER)", pd.to_datetime('2026-05-30'))
fecha_fin_input = st.sidebar.date_input("Fecha Fin del Análisis", pd.to_datetime('2026-07-12'))

# --- MOTOR PRINCIPAL ---
# Solo ejecutamos si el usuario subió ambos archivos
if ops_file is not None and audit_file is not None:
    with st.spinner('Procesando datos y generando reporte...'):
        
        # 1. Lectura
        df_ops = pd.read_csv(ops_file)
        df_audit = pd.read_csv(audit_file)

        df_ops.fillna(0, inplace=True)
        df_ops['report_date'] = pd.to_datetime(df_ops['report_date'])
        df_audit['fecha_modificacion'] = pd.to_datetime(df_audit['fecha_modificacion'])
        df_ops['dia_semana'] = df_ops['report_date'].dt.day_name()

        # 2. Resumen de Mapa
        df_mapa = df_audit.copy()
        df_mapa['fecha_str'] = df_mapa['fecha_modificacion'].dt.strftime('%d-%m-%Y')
        dias_es = {'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mié', 'Thursday': 'Jue', 'Friday': 'Vie', 'Saturday': 'Sáb', 'Sunday': 'Dom'}
        df_mapa['dia_semana'] = df_mapa['dia_semana'].map(dias_es)
        df_mapa_resumen = df_mapa.groupby(['warehouse_name', 'horario_anterior', 'nuevo_horario', 'fecha_str'])['dia_semana'].apply(lambda x: ', '.join(x)).reset_index().sort_values(by='warehouse_name')

        # 3. Lógica de Horas
        def time_to_hour_start(t_str):
            try: return (int(str(t_str).split(':')[0]) + 24) if int(str(t_str).split(':')[0]) < 5 else int(str(t_str).split(':')[0])
            except: return 0

        def time_to_hour_end(t_str):
            try:
                parts = str(t_str).split(':')
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0
                h_adj = h if h >= 5 else h + 24
                return h_adj - 1 if m == 0 else h_adj
            except: return 0

        df_audit['h_ant_num'] = df_audit['horario_anterior'].apply(time_to_hour_start)
        df_audit['h_nue_num'] = df_audit['nuevo_horario'].apply(time_to_hour_end)
        df_audit_clean = df_audit.groupby(['warehouse_name', 'dia_semana']).agg(h_base=('h_ant_num', 'min'), h_final=('h_nue_num', 'max')).reset_index()

        # 4. Cruce y Fechas Dinámicas
        df_master = pd.merge(df_ops, df_audit_clean, on=['warehouse_name', 'dia_semana'], how='inner')
        
        fecha_corte_global = pd.to_datetime(fecha_corte_input)
        fecha_fin = pd.to_datetime(fecha_fin_input)
        
        df_master = df_master[df_master['report_date'] <= fecha_fin]
        df_master['periodo'] = np.where(df_master['report_date'] >= fecha_corte_global, 'AFTER', 'BEFORE')
        df_master['hora_ajustada'] = np.where(df_master['hour'] < 5, df_master['hour'] + 24, df_master['hour'])

        mask_franja = (df_master['hora_ajustada'] >= df_master['h_base']) & (df_master['hora_ajustada'] <= df_master['h_final'])
        df_franja = df_master[mask_franja].copy()

        # 5. Agrupación y Deltas
        df_agrupado = df_franja.groupby(['warehouse_name', 'periodo']).agg(
            dias_operativos=('report_date', 'nunique'), volumen=('orders_completed', 'sum'),
            canceladas=('orders_cancelled', 'sum'), dt_sum=('sum_delivery_time', 'sum'), dt_orders=('dt_orders', 'sum')
        ).reset_index()

        df_agrupado['volumen_diario'] = df_agrupado['volumen'] / df_agrupado['dias_operativos']
        df_agrupado['fail_rate'] = (df_agrupado['canceladas'] / (df_agrupado['volumen'] + df_agrupado['canceladas'])) * 100
        df_agrupado['dt_promedio'] = df_agrupado['dt_sum'] / df_agrupado['dt_orders']
        df_agrupado.fillna(0, inplace=True)

        df_pivot_raw = df_agrupado.pivot(index='warehouse_name', columns='periodo', values=['volumen_diario', 'fail_rate', 'dt_promedio'])
        df_pivot_raw.columns = [f'{col[0]}_{col[1].lower()}' if col[1] else col[0] for col in df_pivot_raw.columns]
        df_pivot = df_pivot_raw.reset_index()

        for col in ['volumen_diario_before', 'fail_rate_before', 'dt_promedio_before', 'volumen_diario_after', 'fail_rate_after', 'dt_promedio_after']:
            if col not in df_pivot.columns: df_pivot[col] = 0.0

        df_pivot['delta_volumen_diario'] = df_pivot['volumen_diario_after'] - df_pivot['volumen_diario_before']
        df_pivot['delta_fail_rate'] = df_pivot['fail_rate_after'] - df_pivot['fail_rate_before']
        df_pivot['delta_dt'] = df_pivot['dt_promedio_after'] - df_pivot['dt_promedio_before']
        df_pivot['seamless_before'] = 100 - df_pivot['fail_rate_before']
        df_pivot['seamless_after'] = 100 - df_pivot['fail_rate_after']
        df_pivot['delta_seamless'] = df_pivot['seamless_after'] - df_pivot['seamless_before']

        def generar_diagnostico(row):
            vol, fr, delta_fr, dt = row['delta_volumen_diario'], row['fail_rate_after'], row['delta_fail_rate'], row['dt_promedio_after']
            diag_traccion = "🟢 Tracción Positiva" if vol > 1.5 else ("🟡 Tracción Leve" if vol > 0 else "🔴 Sin Tracción")
            if fr >= 8.5 or delta_fr > 3.0: diag_ops = "🔴 Alerta FR"
            elif dt >= 35.0: diag_ops = "🟡 Tiempos Altos"
            else: diag_ops = "🟢 Operación Estable"
            return f"{diag_traccion} | {diag_ops}"

        df_pivot['Diagnostico'] = df_pivot.apply(generar_diagnostico, axis=1)

        def categorizar_cuadrante(row):
            vol, fr = row['delta_volumen_diario'], row['delta_fail_rate']
            if vol > 0 and fr <= 1.0: return "Éxito: Gana Vol, FR Controlado"
            elif vol > 0 and fr > 1.0: return "Fricción: Gana Vol, pero Sube FR"
            elif vol <= 0 and fr <= 1.0: return "Sin Tracción: Pierde Vol, FR Controlado"
            else: return "Alerta: Pierde Vol y Sube FR"

        df_pivot['Cuadrante_Grafico'] = df_pivot.apply(categorizar_cuadrante, axis=1)
        df_pivot = df_pivot.sort_values(by='delta_volumen_diario', ascending=False).reset_index(drop=True)

        # 6. Ejecutivos
        df_after_global = df_agrupado[df_agrupado['periodo'] == 'AFTER']
        total_tiendas = df_pivot['warehouse_name'].nunique()
        total_volumen_after, total_canc_after = df_after_global['volumen'].sum(), df_after_global['canceladas'].sum()
        fr_global = (total_canc_after / (total_volumen_after + total_canc_after)) * 100 if (total_volumen_after + total_canc_after) > 0 else 0
        
        dias_after_periodo = df_after_global['dias_operativos'].max() if not df_after_global.empty else 0
        run_rate_nacional = df_pivot['delta_volumen_diario'].sum()
        volumen_neto_ganado = run_rate_nacional * dias_after_periodo

        cuad_exito = len(df_pivot[df_pivot['Cuadrante_Grafico'] == 'Éxito: Gana Vol, FR Controlado'])
        cuad_friccion = len(df_pivot[df_pivot['Cuadrante_Grafico'] == 'Fricción: Gana Vol, pero Sube FR'])
        cuad_sin_trac = len(df_pivot[df_pivot['Cuadrante_Grafico'] == 'Sin Tracción: Pierde Vol, FR Controlado'])
        cuad_alerta = len(df_pivot[df_pivot['Cuadrante_Grafico'] == 'Alerta: Pierde Vol y Sube FR'])

        # 7. Gráfico Base64
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid", context="talk")
        fig_mat, ax_mat = plt.subplots(figsize=(11, 5.5))
        palette = {"Éxito: Gana Vol, FR Controlado": "#27ae60", "Fricción: Gana Vol, pero Sube FR": "#f39c12", "Sin Tracción: Pierde Vol, FR Controlado": "#95a5a6", "Alerta: Pierde Vol y Sube FR": "#c0392b"}
        sns.scatterplot(data=df_pivot, x='delta_fail_rate', y='delta_volumen_diario', hue='Cuadrante_Grafico', palette=palette, s=150, alpha=0.85, edgecolor='black', ax=ax_mat)
        ax_mat.axvline(x=0, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_mat.axhline(y=0, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_mat.set_title('Matriz de Impacto: Órdenes Incrementales vs Variación de Fail Rate', fontsize=15, weight='bold', pad=15)
        ax_mat.set_xlabel('Variación de Fail Rate (%)', weight='bold', fontsize=12)
        ax_mat.set_ylabel('Órdenes Adicionales por Día', weight='bold', fontsize=12)
        ax_mat.legend(title='Lectura del Cuadrante', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
        sns.despine()
        plt.tight_layout()
        
        buf = BytesIO()
        fig_mat.savefig(buf, format='png', bbox_inches='tight', dpi=100, transparent=True)
        buf.seek(0)
        img_mat_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_mat)

        # 8. HTML Output
        def fmt_1d(val): return f"{round(val, 1):g}%"
        def fmt_val(val): return f"{round(val, 1)}" if pd.notnull(val) else "0.0"
        def get_color_gradient(val, min_val, max_val, is_higher_better=True):
            if pd.isnull(val) or max_val == min_val: return "#7f8c8d"
            norm_val = (val - min_val) / (max_val - min_val)
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#e74c3c", "#95a5a6", "#2ecc71"]) if is_higher_better else mcolors.LinearSegmentedColormap.from_list("custom", ["#2ecc71", "#95a5a6", "#e74c3c"])
            return mcolors.to_hex(cmap(norm_val))

        min_fr, max_fr = df_pivot['delta_fail_rate'].min(), df_pivot['delta_fail_rate'].max()
        min_dt, max_dt = df_pivot['delta_dt'].min(), df_pivot['delta_dt'].max()
        min_sea, max_sea = df_pivot['delta_seamless'].min(), df_pivot['delta_seamless'].max()

        html_ejecutivo = f"""
        <div style="background-color: #fff; border: 1px solid #bdc3c7; border-radius: 8px; padding: 20px; margin-bottom: 35px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background-color: #f8f9fa; border-left: 4px solid #34495e; padding: 15px; border-radius: 4px;">
                    <div style="font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase;">Dmarts Modificados</div>
                    <div style="font-size: 24px; color: #2c3e50; font-weight: 900;">{total_tiendas}</div>
                </div>
                <div style="flex: 1; background-color: #eaf2f8; border-left: 4px solid #2980b9; padding: 15px; border-radius: 4px;">
                    <div style="font-size: 11px; color: #2471a3; font-weight: bold; text-transform: uppercase;">Volumen Neto Ganado</div>
                    <div style="font-size: 24px; color: #21618c; font-weight: 900;">+{int(volumen_neto_ganado):,} <span style="font-size: 12px; color: #5499c7; font-weight: normal;">órdenes puras</span></div>
                </div>
                <div style="flex: 1; background-color: #e8f6f3; border-left: 4px solid #27ae60; padding: 15px; border-radius: 4px;">
                    <div style="font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase;">Run-Rate Incremental</div>
                    <div style="font-size: 24px; color: #27ae60; font-weight: 900;">+{fmt_val(run_rate_nacional)} <span style="font-size: 12px; color: #7f8c8d; font-weight: normal;">órd/día</span></div>
                </div>
                <div style="flex: 1; background-color: #fcf3cf; border-left: 4px solid #f39c12; padding: 15px; border-radius: 4px;">
                    <div style="font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase;">Fail Rate de Madrugada</div>
                    <div style="font-size: 24px; color: #d35400; font-weight: 900;">{fmt_1d(fr_global)}</div>
                </div>
            </div>
        </div>
        """

        table_perf = """<div style="max-height: 420px; overflow-y: auto; border: 1px solid #bdc3c7; border-radius: 4px;"><table style="width: 100%; border-collapse: collapse; font-size: 11px; background-color: #fff;"><thead style="position: sticky; top: 0; background-color: #2980b9; color: #fff; z-index: 1;"><tr><th style="padding: 10px; text-align: left;">Dmart</th><th style="padding: 10px; text-align: center;">Vol Incremental/Día</th><th style="padding: 10px; text-align: center; background-color: #21618c;">FR After</th><th style="padding: 10px; text-align: center; background-color: #21618c;">Var. FR</th><th style="padding: 10px; text-align: center; background-color: #1f618d;">DT After</th><th style="padding: 10px; text-align: center; background-color: #1f618d;">Var. DT</th><th style="padding: 10px; text-align: center; background-color: #1a5276;">Seamless After</th><th style="padding: 10px; text-align: center; background-color: #1a5276;">Var. Seamless</th><th style="padding: 10px; text-align: left;">Diagnóstico Operativo</th></tr></thead><tbody>"""
        for _, row in df_pivot.iterrows():
            vol_val = row['delta_volumen_diario']
            vol_str, vol_style = (f"+{fmt_val(vol_val)}", "color: #27ae60; font-weight: bold;") if vol_val > 0 else (f"{fmt_val(vol_val)}", "color: #c0392b; font-weight: bold;")
            fr_style = "color: #c0392b; font-weight: bold;" if row['fail_rate_after'] >= 8.5 else "color: #2c3e50;"
            
            fr_delta = row['delta_fail_rate']
            fr_delta_str = f"+{fmt_1d(fr_delta)}" if fr_delta > 0 else f"{fmt_1d(fr_delta)}"
            fr_color = get_color_gradient(fr_delta, min_fr, max_fr, is_higher_better=False)
            
            dt_delta = row['delta_dt']
            dt_delta_str = f"+{fmt_val(dt_delta)} min" if dt_delta > 0 else f"{fmt_val(dt_delta)} min"
            dt_color = get_color_gradient(dt_delta, min_dt, max_dt, is_higher_better=False)
            
            sea_delta = row['delta_seamless']
            sea_delta_str = f"+{fmt_val(sea_delta)}%" if sea_delta > 0 else f"{fmt_val(sea_delta)}%"
            sea_color = get_color_gradient(sea_delta, min_sea, max_sea, is_higher_better=True)

            diag_parts = row['Diagnostico'].split(" | ")
            diag_html = f"<span style='font-size: 11px;'><b>{diag_parts[0]}</b><br>{diag_parts[1]}</span>"

            table_perf += f"""<tr style="border-bottom: 1px solid #ecf0f1;"><td style="padding: 8px 10px; font-weight: bold; color: #2c3e50;">{row['warehouse_name']}</td><td style="padding: 8px 10px; text-align: center; {vol_style}">{vol_str}</td><td style="padding: 8px 10px; text-align: center; {fr_style}">{fmt_1d(row['fail_rate_after'])}</td><td style="padding: 8px 10px; text-align: center; font-weight: bold; color: {fr_color};">{fr_delta_str}</td><td style="padding: 8px 10px; text-align: center; color: #2c3e50;">{fmt_val(row['dt_promedio_after'])} min</td><td style="padding: 8px 10px; text-align: center; font-weight: bold; color: {dt_color};">{dt_delta_str}</td><td style="padding: 8px 10px; text-align: center; font-weight: bold; color: #2c3e50;">{fmt_1d(row['seamless_after'])}</td><td style="padding: 8px 10px; text-align: center; font-weight: bold; color: {sea_color};">{sea_delta_str}</td><td style="padding: 8px 10px; text-align: left; color: #34495e;">{diag_html}</td></tr>"""
        table_perf += "</tbody></table></div>"

        html_final = f"""
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style> * {{ font-family: 'Outfit', sans-serif !important; }} </style>
        <div style="width: 100%; max-width: 1200px; margin: 0 auto; padding: 20px; font-family: 'Outfit', sans-serif;">
            {html_ejecutivo}
            <div style="display: flex; justify-content: center; margin-bottom: 25px;">
                <img src="data:image/png;base64,{img_mat_b64}" style="width: 100%; max-height: 400px; object-fit: contain;">
            </div>
            {table_perf}
        </div>
        """
        
        # Renderizamos el HTML directamente en la App
        st.markdown(html_final, unsafe_allow_html=True)

else:
    st.info("👈 Por favor, subí ambos archivos CSV en el panel de la izquierda para comenzar el análisis.")