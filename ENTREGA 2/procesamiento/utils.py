import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Análisis detallado de valores faltantes
def analyze_missing_values(df):
    """Análisis completo de valores faltantes"""
    missing_df = pd.DataFrame({
        'Columna': df.columns,
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje': (df.isnull().sum() / len(df)) * 100,
        'Tipo_Dato': df.dtypes
    })

    missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values(
        'Porcentaje', ascending=False
    )

    if len(missing_df) > 0:
        # Visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico de barras
        ax1.bar(missing_df['Columna'], missing_df['Porcentaje'], color='coral')
        ax1.set_xlabel('Columna')
        ax1.set_ylabel('Porcentaje de Valores Faltantes (%)')
        ax1.set_title('Valores Faltantes por Columna')
        ax1.axhline(y=5, color='r', linestyle='--', label='Umbral 5%')
        ax1.legend()

        # Heatmap de patrones
        import seaborn as sns
        msno_data = df[missing_df['Columna'].tolist()].isnull().astype(int)
        sns.heatmap(msno_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                   ax=ax2, vmin=-1, vmax=1)
        ax2.set_title('Correlación de Patrones de Valores Faltantes')

        plt.tight_layout()
        plt.show()

        return missing_df
    else:
        print("✅ No hay valores faltantes en el dataset")
        return None


# Función para análisis univariado robusto
def univariate_analysis(df, column, target=None):
    """Análisis univariado con estadísticas robustas"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histograma con KDE
    ax1 = axes[0, 0]
    df[column].hist(bins=50, edgecolor='black', alpha=0.7, ax=ax1)
    ax1.axvline(df[column].mean(), color='red', linestyle='--', label=f'Media: {df[column].mean():.2f}')
    ax1.axvline(df[column].median(), color='green', linestyle='--', label=f'Mediana: {df[column].median():.2f}')
    ax1.set_title(f'Distribución de {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frecuencia')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Boxplot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(df[column].dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_title(f'Boxplot de {column}')
    ax2.set_ylabel(column)
    ax2.grid(alpha=0.3)

    # Detectar outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    ax2.text(1.1, Q3, f'Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)',
             fontsize=10)

    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(df[column].dropna(), dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normalidad)')
    ax3.grid(alpha=0.3)

    # 4. Relación con target (si existe)
    ax4 = axes[1, 1]
    if target is not None and target in df.columns:
        ax4.scatter(df[column], df[target], alpha=0.5, s=10)
        ax4.set_xlabel(column)
        ax4.set_ylabel(target)
        ax4.set_title(f'{column} vs {target}')

        # Agregar línea de tendencia
        z = np.polyfit(df[column].dropna(), df[target][df[column].notna()], 1)
        p = np.poly1d(z)
        ax4.plot(df[column].sort_values(), p(df[column].sort_values()),
                "r--", alpha=0.8, label=f'Tendencia')

        # Calcular correlación
        corr = df[column].corr(df[target])
        ax4.text(0.05, 0.95, f'Correlación: {corr:.3f}',
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat'))
        ax4.legend()
    else:
        # Estadísticas adicionales
        ax4.axis('off')
        stats_text = f"""
        Estadísticas Robustas:

        • Media: {df[column].mean():.2f}
        • Mediana: {df[column].median():.2f}
        • Desv. Estándar: {df[column].std():.2f}
        • MAD: {stats.median_abs_deviation(df[column].dropna()):.2f}
        • Asimetría: {df[column].skew():.2f}
        • Curtosis: {df[column].kurtosis():.2f}
        • Rango: [{df[column].min():.2f}, {df[column].max():.2f}]
        • IQR: {IQR:.2f}
        • CV: {df[column].std()/df[column].mean():.2f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    ax4.grid(alpha=0.3)

    plt.suptitle(f'Análisis Univariado: {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Análisis variables categóricas
def analyze_categorical(df, cat_col, target_col):
    """Análisis completo de variable categórica"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribución de categorías
    ax1 = axes[0, 0]
    counts = df[cat_col].value_counts()
    ax1.bar(counts.index, counts.values, color=plt.cm.Set3(range(len(counts))))
    ax1.set_title(f'Distribución de {cat_col}')
    ax1.set_xlabel(cat_col)
    ax1.set_ylabel('Frecuencia')
    ax1.tick_params(axis='x', rotation=45)

    # Agregar porcentajes
    for i, (idx, val) in enumerate(counts.items()):
        ax1.text(i, val, f'{val}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom')

    # 2. Pie chart
    ax2 = axes[0, 1]
    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=plt.cm.Set3(range(len(counts))))
    ax2.set_title(f'Proporción de {cat_col}')

    # 3. Boxplot por categoría
    ax3 = axes[1, 0]
    df.boxplot(column=target_col, by=cat_col, ax=ax3)
    ax3.set_title(f'{target_col} por {cat_col}')
    ax3.set_xlabel(cat_col)
    ax3.set_ylabel(target_col)
    plt.sca(ax3)
    plt.xticks(rotation=45)

    # 4. Estadísticas por categoría
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_by_cat = df.groupby(cat_col)[target_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)

    table_data = []
    for idx, row in stats_by_cat.iterrows():
        table_data.append([idx, f"{row['count']:.0f}",
                          f"${row['mean']:,.0f}",
                          f"${row['median']:,.0f}",
                          f"${row['std']:,.0f}"])

    table = ax4.table(cellText=table_data,
                     colLabels=['Categoría', 'N', 'Media', 'Mediana', 'Desv.Est.'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Colorear encabezados
    for i in range(5):
        table[(0, i)].set_facecolor('#40E0D0')
        table[(0, i)].set_text_props(weight='bold')

    plt.suptitle(f'Análisis de Variable Categórica: {cat_col}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def correlation_analysis(df: pd.DataFrame, target: str, threshold: float = 0.10):
    df_num = df.select_dtypes(include=[np.number]).copy()

    # Asegura que el target esté en numérico si es binario no numérico (e.g., Yes/No)
    if target in df.columns and target not in df_num.columns:
        uniques = df[target].dropna().unique()
        if len(uniques) == 2:
            # Mapea los dos valores en 0/1 de forma estable
            mapping = {uniques[0]: 0, uniques[1]: 1}
            df_num[target] = df[target].map(mapping)
            print(f"[info] '{target}' no era numérica. Convertida a 0/1 con mapping: {mapping}")
        else:
            raise ValueError(
                f"El target '{target}' no es numérico ni binario. "
                "Convierte/encodea manualmente antes de correr este análisis."
            )
    elif target not in df_num.columns:
        raise ValueError(f"El target '{target}' no está en las columnas numéricas del DataFrame.")

    numeric_cols = df_num.columns.tolist()

    # --- 1) Pearson ---
    corr_pearson = df_num[numeric_cols].corr(method='pearson')
    # Triangular inferior (ocultar triángulo superior)
    mask_upper = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
    z_pearson = corr_pearson.mask(mask_upper)

    fig1 = go.Figure(
        data=go.Heatmap(
            z=z_pearson.values,
            x=z_pearson.columns,
            y=z_pearson.index,
            colorscale='RdBu',
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title='r'),
            hovertemplate="x: %{x}<br>y: %{y}<br>r: %{z:.2f}<extra></extra>"
        )
    )
    fig1.update_layout(
        title="Correlación de Pearson",
        xaxis=dict(tickangle=45)
    )
    fig1.show()

    # --- 2) Spearman ---
    corr_spearman = df_num[numeric_cols].corr(method='spearman')
    z_spearman = corr_spearman.mask(mask_upper)

    fig2 = go.Figure(
        data=go.Heatmap(
            z=z_spearman.values,
            x=z_spearman.columns,
            y=z_spearman.index,
            colorscale='RdBu',
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title='ρ'),
            hovertemplate="x: %{x}<br>y: %{y}<br>ρ: %{z:.2f}<extra></extra>"
        )
    )
    fig2.update_layout(
        title="Correlación de Spearman",
        xaxis=dict(tickangle=45)
    )
    fig2.show()

    # --- 3) Correlación respecto al target ---
    target_series = corr_pearson[target].drop(labels=[target]).sort_values(ascending=True)
    colors = np.where(target_series.values > 0, "green", "red")

    fig3 = go.Figure(
        go.Bar(
            x=target_series.values,
            y=target_series.index,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>"
        )
    )
    fig3.add_vline(x=0, line_width=1, line_dash="solid")
    fig3.update_layout(
        title=f"Correlación con '{target}' (Pearson)",
        xaxis_title="Coeficiente de correlación",
        yaxis_title="Variable",
        bargap=0.2
    )
    fig3.show()

    # --- Resumen de correlaciones "significativas" ---
    print("\n🔗 Correlaciones Significativas con el Target:")
    print("=" * 60)
    significant = target_series[abs(target_series) > threshold]
    for var, corr in significant.items():
        strength = ("Fuerte" if abs(corr) > 0.5
                    else "Moderada" if abs(corr) > 0.3
                    else "Débil")
        direction = "Positiva" if corr > 0 else "Negativa"
        print(f"  • {var:25s}: {corr:+.3f} ({strength} {direction})")

    return {"pearson": fig1, "spearman": fig2, "target_corr": fig3}
  
def detect_outliers(df):
    """Detección de outliers usando múltiples métodos"""

    numeric_df = df.select_dtypes(include=[np.number])

    # Método 1: IQR
    outliers_iqr = pd.DataFrame()
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df[col] < Q1 - 1.5 * IQR) |
                    (numeric_df[col] > Q3 + 1.5 * IQR))
        outliers_iqr[col] = outliers

    # Método 2: Z-Score
    from scipy import stats
    z_scores = np.abs(stats.zscore(numeric_df.fillna(numeric_df.median())))
    outliers_zscore = (z_scores > 3)

    # Método 3: Isolation Forest
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.median()))
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso_forest.fit_predict(scaled_data) == -1

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Outliers por columna (IQR)
    ax1 = axes[0, 0]
    outlier_counts = outliers_iqr.sum()
    ax1.bar(range(len(outlier_counts)), outlier_counts.values)
    ax1.set_xticks(range(len(outlier_counts)))
    ax1.set_xticklabels(outlier_counts.index, rotation=45, ha='right')
    ax1.set_title('Outliers por Variable (Método IQR)')
    ax1.set_ylabel('Número de Outliers')

    # Plot 2: Distribución de outliers por método
    ax2 = axes[0, 1]
    methods_comparison = pd.DataFrame({
        'IQR': outliers_iqr.any(axis=1).sum(),
        'Z-Score': outliers_zscore.any(axis=1).sum(),
        'Isolation Forest': outliers_iso.sum()
    }, index=['Outliers'])
    methods_comparison.T.plot(kind='bar', ax=ax2, legend=False)
    ax2.set_title('Comparación de Métodos de Detección')
    ax2.set_ylabel('Número de Outliers Detectados')
    ax2.set_xlabel('Método')

    # Plot 3: Heatmap de outliers
    ax3 = axes[1, 0]
    sample_outliers = outliers_iqr.head(100)
    sns.heatmap(sample_outliers.T, cmap='RdYlBu_r', cbar=False, ax=ax3,
                yticklabels=True, xticklabels=False)
    ax3.set_title('Mapa de Outliers (Primeras 100 filas)')
    ax3.set_xlabel('Observaciones')

    # Plot 4: Resumen estadístico
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Resumen de Detección de Anomalías:

    • Total de observaciones: {len(df):,}
    • Outliers por IQR: {outliers_iqr.any(axis=1).sum():,} ({outliers_iqr.any(axis=1).sum()/len(df)*100:.1f}%)
    • Outliers por Z-Score: {outliers_zscore.any(axis=1).sum():,} ({outliers_zscore.any(axis=1).sum()/len(df)*100:.1f}%)
    • Outliers por Isolation Forest: {outliers_iso.sum():,} ({outliers_iso.sum()/len(df)*100:.1f}%)

    Variables más afectadas:
    {chr(10).join([f'  - {col}: {count:,} outliers'
                    for col, count in outlier_counts.nlargest(3).items()])}

    Recomendación: Investigar outliers antes de eliminar.
    Pueden contener información valiosa.
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Análisis de Outliers y Anomalías', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return outliers_iqr, outliers_zscore, outliers_iso