import boto3
import pandas as pd
import openai
import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

# Configurações de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecs_analysis.log'),
        logging.StreamHandler()
    ]
)

# Configurações da API OpenAI
openai.api_key = 'SUA_CHAVE_API_OPENAI'

# Função para coletar dados do ECS com tratamento de exceções
def collect_ecs_data():
    cache_file = 'ecs_data_cache.pkl'

    # Verificar se o cache existe
    if os.path.exists(cache_file):
        logging.info("Carregando dados do cache local.")
        with open(cache_file, 'rb') as f:
            ecs_data = pickle.load(f)
    else:
        logging.info("Cache não encontrado. Coletando dados dos recursos ECS...")
        ecs_client = boto3.client('ecs')
        cloudwatch_client = boto3.client('cloudwatch')

        ecs_data = []

        try:
            clusters = ecs_client.list_clusters()['clusterArns']
        except Exception as e:
            logging.error(f"Erro ao listar clusters ECS: {e}")
            return []

        for cluster_arn in clusters:
            cluster_name = cluster_arn.split('/')[-1]
            try:
                services = ecs_client.list_services(cluster=cluster_arn)['serviceArns']
            except Exception as e:
                logging.error(f"Erro ao listar serviços no cluster {cluster_name}: {e}")
                continue

            for service_arn in services:
                service_name = service_arn.split('/')[-1]
                try:
                    service_desc = ecs_client.describe_services(
                        cluster=cluster_arn,
                        services=[service_name]
                    )['services'][0]
                except Exception as e:
                    logging.error(f"Erro ao descrever o serviço {service_name}: {e}")
                    continue

                task_definition = service_desc['taskDefinition']
                try:
                    task_desc = ecs_client.describe_task_definition(
                        taskDefinition=task_definition
                    )['taskDefinition']
                except Exception as e:
                    logging.error(f"Erro ao descrever a task definition {task_definition}: {e}")
                    continue

                # Coletar métricas de CPU e memória do CloudWatch
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=7)  # Últimos 7 dias

                try:
                    cpu_metrics = cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/ECS',
                        MetricName='CPUUtilization',
                        Dimensions=[
                            {'Name': 'ClusterName', 'Value': cluster_name},
                            {'Name': 'ServiceName', 'Value': service_name},
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,  # Dados diários
                        Statistics=['Average', 'Maximum']
                    )
                except Exception as e:
                    logging.error(f"Erro ao obter métricas de CPU para o serviço {service_name}: {e}")
                    cpu_metrics = {'Datapoints': []}

                try:
                    memory_metrics = cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/ECS',
                        MetricName='MemoryUtilization',
                        Dimensions=[
                            {'Name': 'ClusterName', 'Value': cluster_name},
                            {'Name': 'ServiceName', 'Value': service_name},
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=['Average', 'Maximum']
                    )
                except Exception as e:
                    logging.error(f"Erro ao obter métricas de memória para o serviço {service_name}: {e}")
                    memory_metrics = {'Datapoints': []}

                ecs_data.append({
                    'ClusterName': cluster_name,
                    'ServiceName': service_name,
                    'DesiredCount': service_desc.get('desiredCount', 'N/A'),
                    'RunningCount': service_desc.get('runningCount', 'N/A'),
                    'CPUReservation': service_desc.get('cpu', 'N/A'),
                    'MemoryReservation': service_desc.get('memory', 'N/A'),
                    'TaskDefinition': task_definition,
                    'CPUUtilizationAvg': cpu_metrics['Datapoints'],
                    'MemoryUtilizationAvg': memory_metrics['Datapoints'],
                    # Adicione outros dados conforme necessário
                })

        # Salvar dados no cache
        with open(cache_file, 'wb') as f:
            pickle.dump(ecs_data, f)
            logging.info("Dados salvos no cache local.")

    return ecs_data

# Função para estruturar os dados
def structure_data(ecs_data):
    df = pd.DataFrame(ecs_data)

    # Função auxiliar para extrair o valor médio
    def extract_avg(datapoints):
        if datapoints:
            return sum(dp['Average'] for dp in datapoints) / len(datapoints)
        else:
            return None

    df['CPUUtilizationAvg'] = df['CPUUtilizationAvg'].apply(extract_avg)
    df['MemoryUtilizationAvg'] = df['MemoryUtilizationAvg'].apply(extract_avg)

    return df

# Função para analisar um recurso com tratamento de exceções
def analyze_resource(resource):
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
    - MemoryUtilizationAvg: {MemoryUtilizationAvg}

    Forneça uma análise detalhada e sugestões específicas para otimização de escalonamento e custos.
    """

    prompt = PromptTemplate(
        input_variables=[
            'ClusterName', 'ServiceName', 'DesiredCount', 'RunningCount',
            'CPUReservation', 'MemoryReservation', 'CPUUtilizationAvg', 'MemoryUtilizationAvg'
        ],
        template=template,
    )

    try:
        llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)
        analysis = chain.run(resource)
    except Exception as e:
        logging.error(f"Erro ao analisar o recurso {resource['ServiceName']}: {e}")
        analysis = "Análise não disponível devido a um erro."

    return analysis

# Função para salvar os resultados em um arquivo Excel
def save_to_excel(dataframe):
    output_file = 'ecs_analysis_output.xlsx'
    try:
        dataframe.to_excel(output_file, index=False)
        logging.info(f"Resultados salvos no arquivo {output_file}.")
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo Excel: {e}")

# Função principal com tratamento de exceções e logging
def main():
    logging.info("Iniciando o processo de coleta e análise dos dados ECS.")

    ecs_data = collect_ecs_data()
    if not ecs_data:
        logging.error("Nenhum dado ECS foi coletado. Encerrando o programa.")
        return

    logging.info("Estruturando os dados coletados.")
    df = structure_data(ecs_data)

    logging.info("Analisando cada recurso com LangChain e OpenAI API.")
    analyses = []
    for index, row in df.iterrows():
        resource = row.dropna().to_dict()  # Remover valores NaN
        logging.info(f"Analisando o recurso {resource.get('ServiceName', 'N/A')} no cluster {resource.get('ClusterName', 'N/A')}.")
        analysis = analyze_resource(resource)
        analyses.append(analysis)

    df['Analysis'] = analyses

    logging.info("Salvando os resultados no arquivo Excel.")
    save_to_excel(df)

    logging.info("Processo concluído com sucesso.")

if __name__ == "__main__":
    main()
