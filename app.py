import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Visualizador de Resultados",
    page_icon="📊",
    layout="wide"
)


def display_interview(interview):
    """Muestra los detalles de una entrevista."""
    with st.expander(f"📋 {interview.get('interview_id', 'Entrevista sin ID')}", expanded=False):
        st.subheader("Códigos Asignados")
        if 'assigned_codes' in interview and interview['assigned_codes']:
            cols = st.columns(3)
            for i, code in enumerate(interview['assigned_codes']):
                cols[i % 3].info(code)
        
        if 'analysis' in interview and interview['analysis']:
            st.subheader("Análisis")
            st.write(interview['analysis'])
        
        if 'validation' in interview and interview['validation']:
            st.subheader("Validación")
            st.write(interview['validation'])

def main():
    st.title("📊 Visualizador de Resultados de Entrevistas")
    st.markdown("---")

    # Intentar cargar el archivo por defecto si existe
    default_file = Path("data/results/combined_coding_results_20250701_015904.json")
    if default_file.exists():
        with open(default_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if data:
        # Crear DataFrame para mejor visualización
        df = pd.DataFrame(data)
        
        # Mostrar estadísticas
        st.sidebar.markdown("### 📊 Estadísticas")
        st.sidebar.metric("Total de Entrevistas", len(df))
        
        # Filtros en la barra lateral
        st.sidebar.markdown("### 🔍 Filtros")
        
        # Filtro por códigos
        all_codes = set()
        for codes in df['assigned_codes']:
            all_codes.update(codes)
        
        selected_codes = st.sidebar.multiselect(
            "Filtrar por códigos:",
            sorted(list(all_codes)),
            key="code_filter"
        )
        
        # Aplicar filtros
        if selected_codes:
            mask = df['assigned_codes'].apply(
                lambda x: any(code in x for code in selected_codes)
            )
            df = df[mask]
        
        # Mostrar detalles de cada entrevista
        st.subheader("🔍 Detalles de las Entrevistas")
        for _, row in df.iterrows():
            display_interview(row)
    else:
        st.warning("No se encontró ningún archivo JSON válido.")

if __name__ == "__main__":
    main()
