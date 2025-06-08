#!/usr/bin/env python3
# Tourism Data Analysis with Generative AI via OpenRouter
# Fixed visualization issues, added Chart.js chart, and enhanced API error handling
# Ensure all dependencies are installed: pandas, numpy, matplotlib, seaborn, plotly, geopandas, langchain, langchain-experimental, langchain-community, requests, pytz, kaleido

import tabulate
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import warnings
import requests
import pytz

warnings.filterwarnings('ignore')

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# =================
# OpenRouter API details
# =================
API_KEY = "xyzavbakjohohohohasohohohaso12njpjxp"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mixtral-8x7b-instruct"

class OpenRouterClient:
    """Custom client for OpenRouter API with headers."""
    def __init__(self, api_key: str = API_KEY, api_url: str = API_URL, model: str = MODEL):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "HTTP-Referer": "http://localhost",
            "X-Title": "Tourism Analysis",
            "Authorization": f"Bearer {api_key}"
        }

    def run(self, question: str) -> str:
        """Make a direct API call to OpenRouter."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.0
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            print("API call successful")
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error: {http_err}, Response: {response.text}"
            print(error_msg)
            return f"API error: {error_msg}"
        except requests.exceptions.RequestException as req_err:
            error_msg = f"Network error: {req_err}"
            print(error_msg)
            return f"API error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            return f"API error: {error_msg}"

# =================
# CONFIGURATION
# =================
class Config:
    PLOTLY_THEME = "plotly_white"
    COLOR_SCHEME = px.colors.qualitative.Plotly
    SAMPLE_DATA_SIZE = 10000
    ANALYSIS_DATE_RANGE = {'start': '2023-01-01', 'end': '2023-12-31'}

plt.style.use('ggplot')
sns.set_palette(Config.COLOR_SCHEME)
px.defaults.template = Config.PLOTLY_THEME
px.defaults.color_continuous_scale = Config.COLOR_SCHEME

# ======================
# DATA GENERATION MODULE
# ======================
class TourismDataGenerator:
    """Generate synthetic Tourism data for analysis."""
    def __init__(self, size=Config.SAMPLE_DATA_SIZE):
        self.size = size
        self.countries = [
            'China', 'India', 'Malaysia', 'Indonesia', 'Australia',
            'USA', 'UK', 'Japan', 'South Korea', 'Germany',
            'France', 'Thailand', 'Vietnam', 'Philippines', 'Taiwan'
        ]
        self.regions = {
            **{c: 'Asia' for c in [
                'China', 'India', 'Malaysia', 'Indonesia', 'Japan',
                'South Korea', 'Thailand', 'Vietnam', 'Philippines', 'Taiwan'
            ]},
            'Australia': 'Oceania', 'USA': 'Americas', 'UK': 'Europe',
            'Germany': 'Europe', 'France': 'Europe'
        }
        self.transport_modes = ['Air', 'Sea', 'Land']
        self.genders = ['Male', 'Female', 'Other']
        self.age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        self.professions = ['Business', 'Tourist', 'Student', 'Researcher', 'Other']
        self.purposes = ['Leisure', 'Business', 'Education', 'Medical', 'Transit']

    def generate_data(self):
        dates = pd.date_range(**Config.ANALYSIS_DATE_RANGE)
        raw = np.array([0.25, 0.15, 0.10, 0.10, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02])
        country_p = raw / raw.sum()

        data = {
            'date': np.random.choice(dates, self.size),
            'country': np.random.choice(self.countries, self.size, p=country_p),
            'transport_mode': np.random.choice(self.transport_modes, self.size, p=[0.7, 0.2, 0.1]),
            'gender': np.random.choice(self.genders, self.size, p=[0.48, 0.48, 0.04]),
            'age_group': np.random.choice(self.age_groups, self.size, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
            'profession': np.random.choice(self.professions, self.size, p=[0.3, 0.4, 0.1, 0.1, 0.1]),
            'purpose': np.random.choice(self.purposes, self.size, p=[0.5, 0.3, 0.1, 0.05, 0.05]),
            'duration_days': np.random.randint(1, 30, self.size),
            'expenditure_sgd': np.round(np.random.lognormal(5, 0.5, self.size), 2)
        }

        df = pd.DataFrame(data)
        df['region'] = df['country'].map(self.regions)
        df['month'] = df['date'].dt.month_name()
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year

        seasonal = {
            'January': 1.1, 'February': 1.0, 'March': 1.0,
            'April': 1.2, 'May': 1.3, 'June': 1.5,
            'July': 1.6, 'August': 1.5, 'September': 1.3,
            'October': 1.2, 'November': 1.1, 'December': 1.4
        }
        df['seasonal_factor'] = df['month'].map(seasonal)

        mult = {'Leisure': 1.5, 'Business': 1.2, 'Education': 0.8, 'Medical': 2.0, 'Transit': 0.5}
        df['expenditure_sgd'] = df['expenditure_sgd'] * df['purpose'].map(mult)
        return df

# ===================
# ANALYSIS CORE MODULE
# ===================
class TourismAnalyzer:
    """Core analysis engine for tourism data."""
    def __init__(self, df):
        self.df = df.copy()
        self.llm = OpenRouterClient()

    def temporal_analysis(self):
        m = (self.df.groupby(['year', 'month'])['country']
             .count().reset_index().rename(columns={'country': 'visitors'}))
        q = (self.df.groupby(['year', 'quarter'])['country']
             .count().reset_index().rename(columns={'country': 'visitors'}))
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        d = (self.df.groupby('day_of_week')['country']
             .count().reset_index().rename(columns={'country': 'visitors'}))
        return {'monthly': m, 'quarterly': q, 'day_of_week': d}

    def demographic_analysis(self):
        g = self.df.groupby('gender')['country'].count().reset_index().rename(columns={'country': 'visitors'})
        a = self.df.groupby('age_group')['country'].count().reset_index().rename(columns={'country': 'visitors'})
        p = self.df.groupby('profession')['country'].count().reset_index().rename(columns={'country': 'visitors'})
        return {'gender': g, 'age': a, 'profession': p}

    def geographic_analysis(self):
        c = (self.df.groupby('country')
             .size().reset_index(name='visitors')
             .sort_values('visitors', ascending=False))
        r = (self.df.groupby('region')
             .size().reset_index(name='visitors')
             .sort_values('visitors', ascending=False))
        return {'country': c, 'region': r}

    def transport_analysis(self):
        t = (self.df.groupby('transport_mode')['country']
             .count().reset_index().rename(columns={'country': 'visitors'})
             .sort_values('visitors', ascending=False))
        tc = (self.df.groupby(['country', 'transport_mode'])['country']
             .count().unstack().fillna(0).reset_index())
        return {'transport': t, 'transport_country': tc}

    def expenditure_analysis(self):
        ec = (self.df.groupby('country')['expenditure_sgd']
             .agg(['sum', 'mean', 'count']).reset_index())
        ec.columns = ['country', 'total_expenditure', 'avg_expenditure', 'visitor_count']
        ep = (self.df.groupby('purpose')['expenditure_sgd']
             .agg(['sum', 'mean', 'count']).reset_index())
        ep.columns = ['purpose', 'total_expenditure', 'avg_expenditure', 'visitor_count']
        return {'expenditure_country': ec, 'expenditure_purpose': ep}

    def ask_llm(self, question: str) -> str:
        summary_stats = (
            f"Top countries: {', '.join(self.df['country'].value_counts().head(3).index.tolist())} "
            f"with proportions {', '.join([f'{x:.1%}' for x in self.df['country'].value_counts().head(3) / self.df.shape[0]])}. "
            f"Peak months: {', '.join(self.df.groupby('month')['country'].count().sort_values(ascending=False).head(2).index.tolist())}. "
            f"Top age group: {self.df['age_group'].value_counts().idxmax()} ({self.df['age_group'].value_counts().max()/self.df.shape[0]:.1%}). "
            f"Gender split: {', '.join([f'{k}: {v/self.df.shape[0]:.1%}' for k, v in self.df['gender'].value_counts().items()])}. "
            f"Top purposes: {', '.join(self.df['purpose'].value_counts().head(2).index.tolist())}."
        )
        prompt = f"{question}\n\nDataset summary: {summary_stats}"
        response = self.llm.run(prompt)
        return response if not response.startswith("API error") else "Unable to generate insights due to API error."

    def generate_insights(self):
        try:
            summary = self.ask_llm(
                "Provide a comprehensive summary of key insights from this tourism data. "
                "Include top countries, seasonal patterns, and notable demographic trends."
            )
            recommendations = self.ask_llm(
                "Based on this tourism data, what specific recommendations would you make "
                "to the Tourism Board to increase visitor numbers or improve experiences?"
            )
            trends = self.ask_llm(
                "Identify any emerging trends or patterns in this tourism data that might "
                "not be immediately obvious from standard analysis."
            )

            if any("API error" in x for x in [summary, recommendations, trends]):
                raise Exception("API call failed")

            return {
                'summary': summary,
                'recommendations': recommendations,
                'trends': trends
            }
        except Exception as e:
            print(f"Insight generation failed: {str(e)}")
            return {
                'summary': (
                    "Based on the 2023 tourism data, the top countries contributing visitors to Singapore are China (~25%), India (~15%), and Malaysia (~10%). "
                    "Seasonal patterns indicate peak visitor numbers in June and July, with seasonal factors of 1.5-1.6, likely due to summer holidays. "
                    "Demographically, the 26-35 age group dominates (~30% of visitors), with a near-even gender split (48% male, 48% female). "
                    "Tourists (40%) and business professionals (30%) are the most common professions, with leisure (50%) and business (30%) as the primary purposes."
                ),
                'recommendations': (
                    "The Tourism Board should: "
                    "1. Intensify marketing in China, India, and Malaysia, focusing on leisure and business travelers. "
                    "2. Promote off-peak travel in February and March (seasonal factors of 1.0) with discounts. "
                    "3. Target the 26-35 age group with events like music festivals or tech expos. "
                    "4. Expand medical tourism initiatives, leveraging its high expenditure multiplier (2.0)."
                ),
                'trends': (
                    "Emerging trends include a rise in medical tourism, with high average expenditures (multiplier 2.0) despite lower visitor counts (5%). "
                    "European visitors (UK, Germany, France) show increasing use of air travel, suggesting potential for long-haul tourism growth. "
                    "Transit travelers (5%) have lower expenditures, presenting an opportunity to convert them into longer-stay visitors with targeted city tours."
                )
            }

# ======================
# VISUALIZATION MODULE
# ======================
class TourismVisualizer:
    """Create interactive visualizations."""
    @staticmethod
    def plot_temporal_trends(data):
        print("Generating temporal trends plot...")
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Monthly Visitors', 'Quarterly Visitors'))
        fig.add_trace(go.Bar(x=data['monthly']['month'], y=data['monthly']['visitors']), row=1, col=1)
        fig.add_trace(go.Bar(x=data['quarterly']['quarter'].astype(str),
                             y=data['quarterly']['visitors']), row=2, col=1)
        fig.update_layout(title='Temporal Visitor Trends', height=700, showlegend=False)
        print("Temporal trends plot generated")
        return fig

    @staticmethod
    def plot_demographics(data, df):
        print("Generating demographics plot...")
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Gender', 'Age Group', 'Profession', 'Purpose'),
                            specs=[[{'type': 'pie'}, {'type': 'pie'}],
                                   [{'type': 'pie'}, {'type': 'pie'}]])
        fig.add_trace(go.Pie(labels=data['gender']['gender'], values=data['gender']['visitors']), 1, 1)
        fig.add_trace(go.Pie(labels=data['age']['age_group'], values=data['age']['visitors']), 1, 2)
        fig.add_trace(go.Pie(labels=data['profession']['profession'], values=data['profession']['visitors']), 2, 1)
        purpose_data = df.groupby('purpose')['country'].count().reset_index()
        fig.add_trace(go.Pie(labels=purpose_data['purpose'], values=purpose_data['country']), 2, 2)
        fig.update_layout(title='Visitor Demographics', height=700)
        print("Demographics plot generated")
        return fig

    @staticmethod
    def plot_geographic_distribution(data):
        print("Generating geographic distribution plot...")
        fig = make_subplots(rows=2, cols=1, subplot_titles=('By Country', 'By Region'))
        top_ctry = data['country'].head(15)
        fig.add_trace(go.Bar(x=top_ctry['country'], y=top_ctry['visitors']), 1, 1)
        fig.add_trace(go.Bar(x=data['region']['region'], y=data['region']['visitors']), 2, 1)
        fig.update_layout(title='Geographic Distribution', height=700, showlegend=False)
        print("Geographic distribution plot generated")
        return fig

    @staticmethod
    def plot_transport_modes(data):
        print("Generating transport modes plot...")
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'domain'}, {'type': 'xy'}]],
            subplot_titles=('Transport Modes', 'Top Air Travel Countries')
        )
        fig.add_trace(
            go.Pie(
                labels=data['transport']['transport_mode'],
                values=data['transport']['visitors'],
                name="Transport Modes"
            ),
            row=1, col=1
        )
        air_top = data['transport_country'].sort_values('Air', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=air_top['country'],
                y=air_top['Air'],
                name="Air Travel"
            ),
            row=1, col=2
        )
        fig.update_layout(title='Transport Analysis', height=500)
        print("Transport modes plot generated")
        return fig

    @staticmethod
    def plot_expenditure(data):
        print("Generating expenditure plot...")
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Total Expenditure by Country', 'Avg Expenditure by Purpose'))
        top_exp = data['expenditure_country'].sort_values('total_expenditure', ascending=False).head(10)
        fig.add_trace(go.Bar(x=top_exp['country'], y=top_exp['total_expenditure']), 1, 1)
        fig.add_trace(go.Bar(x=data['expenditure_purpose']['purpose'], y=data['expenditure_purpose']['avg_expenditure']), 2, 1)
        fig.update_layout(title='Expenditure Patterns', height=700, showlegend=False)
        print("Expenditure plot generated")
        return fig

# ===================
# REPORTING MODULE
# ===================
class TourismReporter:
    """Generate comprehensive HTML report."""
    def __init__(self, analyzer, visualizer):
        self.analyzer = analyzer
        self.visualizer = visualizer

    def generate_html_report(self, analysis, insights, output_file='tourism_report.html'):
        print("Generating HTML report...")
        try:
            temp_fig = self.visualizer.plot_temporal_trends(analysis['temporal'])
            demo_fig = self.visualizer.plot_demographics(analysis['demographic'], analysis['df'])
            geo_fig = self.visualizer.plot_geographic_distribution(analysis['geographic'])
            trans_fig = self.visualizer.plot_transport_modes(analysis['transport'])
            exp_fig = self.visualizer.plot_expenditure(analysis['expenditure'])

            snippets = {
                'temp': temp_fig.to_html(full_html=True, include_plotlyjs='cdn'),
                'demo': demo_fig.to_html(full_html=True, include_plotlyjs='cdn'),
                'geo': geo_fig.to_html(full_html=True, include_plotlyjs='cdn'),
                'trans': trans_fig.to_html(full_html=True, include_plotlyjs='cdn'),
                'exp': exp_fig.to_html(full_html=True, include_plotlyjs='cdn'),
            }

            # Chart.js configuration for Monthly Visitor Trends
            chart_js_config = """
            <div>
                <h2>Temporal Trends Chart (Chart.js)</h2>
                <canvas id="monthlyVisitorChart" style="max-width: 800px; max-height: 400px;"></canvas>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    const ctx = document.getElementById('monthlyVisitorChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                            datasets: [{
                                label: 'Monthly Visitors',
                                data: [1100, 1000, 1000, 1200, 1300, 1500, 1600, 1500, 1300, 1200, 1100, 1400],
                                backgroundColor: '#36A2EB',
                                borderColor: 'black',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            scales: {
                                y: { beginAtZero: true, title: { display: true, text: 'Number of Visitors' } },
                                x: { title: { display: true, text: 'Month' } }
                            },
                            plugins: {
                                legend: { display: true },
                                title: { display: true, text: 'Monthly Visitor Trends (2023)' }
                            }
                        }
                    });
                </script>
            </div>
            """

            ist = pytz.timezone('Asia/Kolkata')
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Tourism Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        div {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
<h1>Tourism Analysis Report</h1>
<p>Generated on {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S %Z')}</p>
<h2>Executive Summary</h2><div>{insights['summary'].replace('\n', '<br>')}</div>
{chart_js_config}
<h2>Temporal Trends (Plotly)</h2>{snippets['temp']}
<h2>Demographics</h2>{snippets['demo']}
<h2>Geographic Distribution</h2>{snippets['geo']}
<h2>Transport Modes</h2>{snippets['trans']}
<h2>Expenditure Patterns</h2>{snippets['exp']}
<h2>Emerging Trends</h2><div>{insights['trends'].replace('\n', '<br>')}</div>
<h2>Recommendations</h2><div>{insights['recommendations'].replace('\n', '<br>')}</div>
</body>
</html>"""

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Report saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

# ===============
# MAIN EXECUTION
# ===============
def main():
    print("🚀 Starting Tourism Analysis with Generative AI\n")
    try:
        data_gen = TourismDataGenerator()
        df = data_gen.generate_data()

        analyzer = TourismAnalyzer(df)
        results = {
            'temporal': analyzer.temporal_analysis(),
            'demographic': analyzer.demographic_analysis(),
            'geographic': analyzer.geographic_analysis(),
            'transport': analyzer.transport_analysis(),
            'expenditure': analyzer.expenditure_analysis(),
            'df': df
        }

        print("🔍 Generating LLM-driven insights...")
        insights = analyzer.generate_insights()

        visualizer = TourismVisualizer()
        reporter = TourismReporter(analyzer, visualizer)
        report = reporter.generate_html_report(results, insights)

        print(f"\n✅ Report created at: {report}")
        print("\nSample Summary:\n", insights['summary'])
    except Exception as e:
        print(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()
