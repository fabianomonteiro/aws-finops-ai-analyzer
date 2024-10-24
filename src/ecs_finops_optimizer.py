# src/ecs_finops_optimizer.py

import boto3
import pandas as pd
import openai
import logging
import os
import json
import pickle
import hashlib
import asyncio
import aiohttp
import nest_asyncio
from datetime import datetime, timedelta
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from boto3.exceptions import Boto3Error
from openai.error import OpenAIError

# Permitir execução de loops aninhados (necessário em alguns ambientes)
nest_asyncio.apply()

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecs_analysis.log'),
        logging.StreamHandler()
    ]
)

# Configuração da API OpenAI
# Substitua 'SUA_CHAVE_API_OPENAI' pela sua chave da API OpenAI ou use uma variável de ambiente
openai.api_key = os.getenv('OPENAI_API_KEY', 'SUA_CHAVE_API_OPENAI')

# Limite máximo de tarefas simultâneas para o semáforo
MAX_CONCURRENT_TASKS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Função para gerar um hash único para cada recurso
def generate_resource_hash(resource):
    resource_string = json.dumps(resource, sort_keys=True)
    return hashlib.md5(resource_string.encode('utf-8')).hexdigest()

# Função para identificar horários de pico e vale
def identify_peak_offpeak_times(row):
    cpu_datapoints = row['CPUUtilizationDataPoints']

    if not cpu_datapoints:
        return [], []

    # Ordenar os datapoints por timestamp
    cpu_datapoints.sort(key=lambda x: x['Timestamp'])

    # Criar um DataFrame temporário para análise
    timestamps = [dp['Timestamp'] for dp in cpu_datapoints]
    cpu_values = [dp['Average'] for dp in cpu_datapoints]

    temp_df = pd.DataFrame({'Timestamp': timestamps, 'CPUUtilization': cpu_values})

    # Converter timestamps para o fuso horário desejado (exemplo: 'America/Sao_Paulo')
    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp']).dt.tz_convert('America/Sao_Paulo')

    # Agrupar por hora para identificar horários de pico
    temp_df['Hour'] = temp_df['Timestamp'].dt.hour

    avg_cpu_by_hour = temp_df.groupby('Hour')['CPUUtilization'].mean()

    # Definir horários de pico como as horas com uso acima de um certo percentil (por exemplo, 75%)
    threshold = avg_cpu_by_hour.quantile(0.75)
    peak_hours = avg_cpu_by_hour[avg_cpu_by_hour >= threshold].index.tolist()
    offpeak_hours = avg_cpu_by_hour[avg_cpu_by_hour < threshold].index.tolist()

    return peak_hours, offpeak_hours

# Função para coletar dados do ECS com métricas detalhadas
def collect_ecs_data():
    cache_file = 'ecs_data_cache.pkl'

    if os.path.exists(cache_file):
        logging.info("Carregando dados do ECS do cache.")
        with open(cache_file, 'rb') as f:
            ecs_data = pickle.load(f)
    else:
        logging.info("Cache não encontrado. Coletando dados do ECS...")
        ecs_client = boto3.client('ecs')
        cloudwatch_client = boto3.client('cloudwatch')
        ce_client = boto3.client('ce')  # Cliente do AWS Cost Explorer
        autoscaling_client = boto3.client('application-autoscaling')

        ecs_data = []

        try:
            clusters = ecs_client.list_clusters()['clusterArns']
        except Boto3Error as e:
            logging.error(f"Erro ao listar clusters ECS: {e}")
            return []

        for cluster_arn in clusters:
            cluster_name = cluster_arn.split('/')[-1]
            try:
                services = ecs_client.list_services(cluster=cluster_arn)['serviceArns']
            except Boto3Error as e:
                logging.error(f"Erro ao listar serviços no cluster {cluster_name}: {e}")
                continue

            for service_arn in services:
                service_name = service_arn.split('/')[-1]
                try:
                    service_desc = ecs_client.describe_services(
                        cluster=cluster_arn,
                        services=[service_name]
                    )['services'][0]
                except Boto3Error as e:
                    logging.error(f"Erro ao descrever o serviço {service_name}: {e}")
                    continue

                task_definition = service_desc['taskDefinition']
                try:
                    task_desc = ecs_client.describe_task_definition(
                        taskDefinition=task_definition
                    )['taskDefinition']
                except Boto3Error as e:
                    logging.error(f"Erro ao descrever a task definition {task_definition}: {e}")
                    continue

                # Definir o período e o intervalo de coleta
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=7)  # Últimos 7 dias
                period = 3600  # Dados horários

                dimensions = [
                    {'Name': 'ClusterName', 'Value': cluster_name},
                    {'Name': 'ServiceName', 'Value': service_name},
                ]

                # Coletar métricas de CPU com granularidade horária
                try:
                    cpu_metrics = cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/ECS',
                        MetricName='CPUUtilization',
                        Dimensions=dimensions,
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=period,
                        Statistics=['Average', 'Maximum']
                    )
                except Boto3Error as e:
                    logging.error(f"Erro ao obter métricas de CPU para o serviço {service_name}: {e}")
                    cpu_metrics = {'Datapoints': []}

                # Coletar outras métricas similares para memória, rede, etc.

                # Coletar dados de custos
                try:
                    cost_response = ce_client.get_cost_and_usage(
                        TimePeriod={
                            'Start': (end_time - timedelta(days=7)).strftime('%Y-%m-%d'),
                            'End': end_time.strftime('%Y-%m-%d')
                        },
                        Granularity='DAILY',
                        Metrics=['UnblendedCost'],
                        Filter={
                            'Dimensions': {
                                'Key': 'Service',
                                'Values': ['Amazon Elastic Container Service']
                            }
                        },
                        GroupBy=[
                            {
                                'Type': 'DIMENSION',
                                    'Key': 'Service'
                            },
                            {
                                'Type': 'TAG',
                                'Key': 'aws:ecs:clusterName'
                            },
                        ]
                    )
                    # Processar dados de custos para obter o custo total do serviço
                    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in cost_response['ResultsByTime'])
                except Boto3Error as e:
                    logging.error(f"Erro ao obter dados de custos para o serviço {service_name}: {e}")
                    total_cost = None

                # Coletar políticas de escalonamento
                try:
                    scaling_policies = autoscaling_client.describe_scaling_policies(
                        ServiceNamespace='ecs',
                        ResourceId=f'service/{cluster_name}/{service_name}'
                    )['ScalingPolicies']
                except Boto3Error as e:
                    logging.error(f"Erro ao obter políticas de escalonamento para o serviço {service_name}: {e}")
                    scaling_policies = []

                resource = {
                    'ClusterName': cluster_name,
                    'ServiceName': service_name,
                    'DesiredCount': service_desc.get('desiredCount', 'N/A'),
                    'RunningCount': service_desc.get('runningCount', 'N/A'),
                    'CPUReservation': service_desc.get('cpu', 'N/A'),
                    'MemoryReservation': service_desc.get('memory', 'N/A'),
                    'TaskDefinition': task_definition,
                    'CPUUtilizationDataPoints': cpu_metrics['Datapoints'],
                    # Adicione outras métricas conforme necessário
                    'TotalCost': total_cost,
                    'ScalingPolicies': scaling_policies,
                }

                # Gerar hash do recurso
                resource_hash = generate_resource_hash(resource)
                resource['ResourceHash'] = resource_hash

                ecs_data.append(resource)

        # Salvar dados no cache
        with open(cache_file, 'wb') as f:
            pickle.dump(ecs_data, f)
            logging.info("Dados do ECS salvos no cache.")

    return ecs_data

# Função para estruturar os dados
def structure_data(ecs_data):
    df = pd.DataFrame(ecs_data)

    # Calcular médias gerais (opcional)
    def extract_avg(datapoints):
        if datapoints:
            return sum(dp['Average'] for dp in datapoints) / len(datapoints)
        else:
            return None

    df['CPUUtilizationAvg'] = df['CPUUtilizationDataPoints'].apply(extract_avg)

    # Identificar horários de pico e vale
    df['PeakTimes'], df['OffPeakTimes'] = zip(*df.apply(identify_peak_offpeak_times, axis=1))

    # Converter políticas de escalonamento para representação em string
    df['ScalingPolicies'] = df['ScalingPolicies'].apply(lambda x: json.dumps(x))

    # Tratar custo total
    df['TotalCost'] = df['TotalCost'].fillna(0.0)

    return df

# Função assíncrona para analisar um recurso com semáforo
async def analyze_resource(resource, session, analysis_cache):
    resource_hash = resource['ResourceHash']

    # Verificar se a análise já está em cache
    if resource_hash in analysis_cache:
        logging.info(f"Análise em cache encontrada para o recurso {resource['ServiceName']}.")
        return analysis_cache[resource_hash]

    # Preparar o prompt com os dados adicionais
    template = """
    Você é um especialista em AWS e FinOps. Analise os seguintes dados do recurso ECS e forneça recomendações de otimização de custos:

    Dados do Recurso:
    - ClusterName: {ClusterName}
    - ServiceName: {ServiceName}
    - DesiredCount: {DesiredCount}
    - RunningCount: {RunningCount}
    - CPUReservation: {CPUReservation}
    - MemoryReservation: {MemoryReservation}
    - CPUUtilizationAvg: {CPUUtilizationAvg}
    - TotalCost (últimos 7 dias): {TotalCost}
    - ScalingPolicies: {ScalingPolicies}
    - Horários de Pico (Horas do dia): {PeakTimes}
    - Horários de Vale (Horas do dia): {OffPeakTimes}

    Forneça uma análise detalhada e sugestões específicas para otimização de escalonamento e custos, levando em consideração os requisitos de negócio e SLAs. Considere ajustar as políticas de escalonamento com base nos horários de pico e vale identificados.
    """

    prompt = PromptTemplate(
        input_variables=[
            'ClusterName', 'ServiceName', 'DesiredCount', 'RunningCount',
            'CPUReservation', 'MemoryReservation', 'CPUUtilizationAvg',
            'TotalCost', 'ScalingPolicies', 'PeakTimes', 'OffPeakTimes'
        ],
        template=template,
    )

    # Certificar-se de que 'PeakTimes' e 'OffPeakTimes' são strings
    resource['PeakTimes'] = ', '.join(map(str, resource.get('PeakTimes', [])))
    resource['OffPeakTimes'] = ', '.join(map(str, resource.get('OffPeakTimes', [])))

    async with semaphore:
        try:
            llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)
            chain = LLMChain(llm=llm, prompt=prompt)
            analysis = await chain.arun(resource)
            # Armazenar a análise no cache
            analysis_cache[resource_hash] = analysis

        except OpenAIError as e:
            logging.error(f"Erro ao analisar o recurso {resource['ServiceName']}: {e}")
            analysis = "Análise não disponível devido a um erro."

    return analysis

# Função para salvar o cache de análises
def save_analysis_cache(analysis_cache):
    cache_file = 'analysis_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(analysis_cache, f)
    logging.info("Cache de análises salvo.")

# Função para carregar o cache de análises
def load_analysis_cache():
    cache_file = 'analysis_cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            analysis_cache = pickle.load(f)
        logging.info("Cache de análises carregado.")
    else:
        analysis_cache = {}
    return analysis_cache

# Função principal assíncrona
async def main_async():
    logging.info("Iniciando a coleta e análise de dados do ECS.")

    ecs_data = collect_ecs_data()
    if not ecs_data:
        logging.error("Nenhum dado do ECS foi coletado. Encerrando o programa.")
        return

    logging.info("Estruturando os dados coletados.")
    df = structure_data(ecs_data)

    # Carregar cache de análises
    analysis_cache = load_analysis_cache()

    logging.info("Analisando recursos de forma assíncrona com LangChain e API OpenAI.")
    tasks = []
    async with aiohttp.ClientSession() as session:
        for index, row in df.iterrows():
            resource = row.dropna().to_dict()  # Remover valores NaN
            logging.info(f"Preparando a análise para o recurso {resource.get('ServiceName', 'N/A')} no cluster {resource.get('ClusterName', 'N/A')}.")
            task = asyncio.ensure_future(analyze_resource(resource, session, analysis_cache))
            tasks.append(task)

        # Executar tarefas assíncronas com semáforo
        analyses = await asyncio.gather(*tasks)

    df['Analysis'] = analyses

    logging.info("Salvando os resultados no arquivo Excel.")
    save_to_excel(df)

    # Salvar cache de análises
    save_analysis_cache(analysis_cache)

    logging.info("Processo concluído com sucesso.")

# Função para salvar os resultados em um arquivo Excel
def save_to_excel(dataframe):
    output_file = 'ecs_analysis_output.xlsx'
    try:
        dataframe.to_excel(output_file, index=False)
        logging.info(f"Resultados salvos no arquivo {output_file}.")
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo Excel: {e}")

# Função principal
def main():
    # Executar a função assíncrona
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
