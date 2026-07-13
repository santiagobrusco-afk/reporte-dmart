import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN NATIVA DE LA APP ---
st.set_page_config(page_title="Performance Extensiones", page_icon="📊", layout="wide")

# CSS mínimo solo para afinar detalles nativos (nada de HTML estructural)
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Análisis de Performance: Extensiones Horarias")
st.caption("Diagnóstico automatizado sobre el impacto en volumen y calidad operativa de la madrugada.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    ops_file = st.file_uploader("Operaciones (CSV)", type="csv")
    audit_file = st.file_uploader("Auditoría (CSV)", type="csv")
    
    st.divider()
    fecha_corte_input = st.date_input("Inicio del Periodo AFTER", pd.to_datetime('2026-05-30'))
    fecha_fin_input = st.date_input("Fin del Análisis", pd.to_datetime('2026-07-12'))

# --- MOTOR PRINCIPAL ---
if ops_file and audit_file:
    with st.spinner('Procesando modelo de datos...'):
        
        # 1. CARGA DE DATOS
        df_ops = pd.read_csv(ops_file)
        df_audit = pd.read_csv(audit_file)

        df_ops.fillna(0, inplace=True)
        df_ops['report_date'] = pd.to_datetime(df_ops['report_date'])
        df_audit['fecha_modificacion'] = pd.to_datetime(df_audit['fecha_modificacion'])
        df_ops['dia_semana'] = df_ops['report_date'].dt.day_name()

        # 2. RESUMEN DE MAPA
        df_mapa = df_audit.copy()
        df_mapa['fecha_str'] = df_mapa['fecha_modificacion'].dt.strftime('%d-%m-%Y')
        dias_es = {'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mié', 'Thursday': 'Jue', 'Friday': 'Vie', 'Saturday': 'Sáb', 'Sunday': 'Dom'}
        df_mapa['dia_semana'] = df_mapa['dia_semana'].map(dias_es)
        df_mapa_resumen = df_mapa.groupby(['warehouse_name', 'horario_anterior', 'nuevo_horario', 'fecha_str'])['dia_semana'].apply(lambda x: ', '.join(x)).reset_index().sort_values(by='warehouse_name')
        df_mapa_resumen['Cambio de Horario'] = df_mapa_resumen['horario_anterior'] + ' ➔ ' + df_mapa_resumen['nuevo_horario']

        # 3. LÓGICA DE HORAS
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

        df_master = pd.merge(df_ops, df_audit_clean, on=['warehouse_name', 'dia_semana'], how='inner')
        fecha_corte_global = pd.to_datetime(fecha_corte_input)
        fecha_fin = pd.to_datetime(fecha_fin_input)
        
        df_master = df_master[df_master['report_date'] <= fecha_fin]
        df_master['periodo'] = np.where(df_master['report_date'] >= fecha_corte_global, 'AFTER', 'BEFORE')
        df_master['hora_ajustada'] = np.where(df_master['hour'] < 5, df_master['hour'] + 24, df_master['hour'])

        mask_franja = (df_master['hora_ajustada'] >= df_master['h_base']) & (df_master['hora_ajustada'] <= df_master['h_final'])
        df_franja = df_master[mask_franja].copy()

        # 4. AGRUPACIÓN Y DELTAS
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
            else: diag_ops = "🟢 Ops Estable"
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

        # 5. EJECUTIVOS GLOBALES
        df_after_global = df_agrupado[df_agrupado['periodo'] == 'AFTER']
        total_tiendas = df_pivot['warehouse_name'].nunique()
        total_volumen_after = df_after_global['volumen'].sum()
        total_canc_after = df_after_global['canceladas'].sum()
        fr_global = (total_canc_after / (total_volumen_after + total_canc_after)) * 100 if (total_volumen_after + total_canc_after) > 0 else 0
        
        dias_after_periodo = df_after_global['dias_operativos'].max() if not df_after_global.empty else 0
        run_rate_nacional = df_pivot['delta_volumen_diario'].sum()
        volumen_neto_ganado = run_rate_nacional * dias_after_periodo

        # ==============================================================================
        # 6. RENDERIZADO 100% NATIVO DE STREAMLIT
        # ==============================================================================
        
        st.divider()
        
        # --- TARJETAS NATIVAS ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container(border=True):
                st.metric("🏪 Dmarts Modificados", f"{total_tiendas}")
        with col2:
            with st.container(border=True):
                st.metric("📦 Volumen Neto Ganado", f"+{int(volumen_neto_ganado):,} órd")
        with col3:
            with st.container(border=True):
                st.metric("🚀 Run-Rate Incremental", f"+{run_rate_nacional:.1f}", delta="Órdenes extra por día", delta_color="normal")
        with col4:
            with st.container(border=True):
                st.metric("⚠️ Fail Rate Global", f"{fr_global:.1f}%", delta="Promedio en Madrugada", delta_color="off")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- NAVEGACIÓN POR PESTAÑAS ---
        tab_matriz, tab_tabla, tab_mapa = st.tabs(["📈 Matriz de Impacto Visual", "📋 Diagnóstico Granular", "🗺️ Mapeo de Implementaciones"])

        # PESTAÑA 1: MATRIZ
        with tab_matriz:
            col_chart, col_spacer = st.columns([8, 1])
            with col_chart:
                plt.rcParams['font.family'] = 'sans-serif'
                sns.set_theme(style="whitegrid", context="talk")
                fig_mat, ax_mat = plt.subplots(figsize=(10, 5))
                
                # Fondo transparente para adaptación al Dark/Light mode del usuario
                fig_mat.patch.set_alpha(0.0)
                ax_mat.patch.set_alpha(0.0)
                
                palette = {"Éxito: Gana Vol, FR Controlado": "#27ae60", "Fricción: Gana Vol, pero Sube FR": "#f39c12", "Sin Tracción: Pierde Vol, FR Controlado": "#95a5a6", "Alerta: Pierde Vol y Sube FR": "#c0392b"}
                sns.scatterplot(data=df_pivot, x='delta_fail_rate', y='delta_volumen_diario', hue='Cuadrante_Grafico', palette=palette, s=150, alpha=0.85, edgecolor='black', ax=ax_mat)
                
                ax_mat.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                ax_mat.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                ax_mat.set_title('Órdenes Incrementales vs Variación de Fail Rate', fontsize=14, weight='bold', pad=15)
                ax_mat.set_xlabel('Variación de Fail Rate (%)', weight='bold', fontsize=11)
                ax_mat.set_ylabel('Órdenes Adicionales por Día', weight='bold', fontsize=11)
                
                # Estilo de leyenda adaptativo
                legend = ax_mat.legend(title='Lectura del Cuadrante', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=11)
                frame = legend.get_frame()
                frame.set_facecolor('white')
                frame.set_alpha(0.8)
                
                sns.despine()
                st.pyplot(fig_mat)

        # PESTAÑA 2: TABLA INTERACTIVA (PANDAS STYLER)
        with tab_tabla:
            st.caption("Podés hacer clic en los nombres de las columnas para ordenar la tabla, o tocar la lupa arriba a la derecha para buscar.")
            
            df_mostrar = df_pivot[['warehouse_name', 'delta_volumen_diario', 'fail_rate_after', 'delta_fail_rate', 'dt_promedio_after', 'delta_dt', 'seamless_after', 'delta_seamless', 'Diagnostico']].copy()
            df_mostrar.columns = ['Dmart', 'Vol Incremental/Día', 'FR After', 'Var. FR', 'DT After', 'Var. DT', 'Seamless After', 'Var. Seamless', 'Diagnóstico Operativo']

            def color_volumen(val): return f"color: {'#27ae60' if val > 0 else '#c0392b'}; font-weight: bold;"
            def color_fr(val): return f"color: {'#c0392b' if val >= 8.5 else 'inherit'}; font-weight: bold;"

            styled_df = df_mostrar.style\
                .map(color_volumen, subset=['Vol Incremental/Día'])\
                .map(color_fr, subset=['FR After'])\
                .background_gradient(cmap='RdYlGn_r', subset=['Var. FR', 'Var. DT'])\
                .background_gradient(cmap='RdYlGn', subset=['Var. Seamless'])\
                .format({
                    'Vol Incremental/Día': '{:+.1f}', 'FR After': '{:.1f}%', 'Var. FR': '{:+.1f}%',
                    'DT After': '{:.1f} min', 'Var. DT': '{:+.1f} min', 'Seamless After': '{:.1f}%', 'Var. Seamless': '{:+.1f}%'
                })

            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=450)

        # PESTAÑA 3: MAPA DE CONFIGURACIÓN
        with tab_mapa:
            df_mapa_vista = df_mapa_resumen[['warehouse_name', 'dia_semana', 'Cambio de Horario', 'fecha_str']].copy()
            df_mapa_vista.columns = ['Dmart', 'Días Modificados', 'Alteración Horaria', 'Fecha Real de Modificación']
            st.dataframe(df_mapa_vista, use_container_width=True, hide_index=True)

else:
    st.info("👈 Por favor, subí los extractos de **Operaciones** y **Auditoría** en el panel izquierdo para comenzar a trabajar.")
