# AWS FinOps Optimizer

Este projeto automatiza a coleta de dados de recursos do AWS ECS, analisa-os usando modelos GPT da OpenAI via LangChain e gera recomendações para otimização de custos em um arquivo Excel.

## **Recursos**

- Coleta detalhada de dados de recursos do AWS ECS.
- Identifica horários de pico e vale de uso.
- Analisa recursos usando modelos GPT da OpenAI.
- Fornece recomendações para otimização de custos.
- Utiliza cache para reduzir chamadas à API.
- Trata exceções e registra logs do processo.

## **Instalação**

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/aws-finops-optimizer.git
   ```

2. Navegue até o diretório do projeto:

   ```bash
   cd aws-finops-optimizer
   ```

3. Instale os pacotes necessários:

   ```bash
   pip install -r requirements.txt
   ```

## **Configuração**

- **Credenciais AWS**: Certifique-se de que suas credenciais AWS estão configuradas corretamente (por exemplo, no arquivo `~/.aws/credentials` ou através de variáveis de ambiente).
- **Chave da API OpenAI**: Defina sua chave da API OpenAI no arquivo `ecs_finops_optimizer.py` ou use uma variável de ambiente.

## **Uso**

Execute o script principal:

```bash
python src/ecs_finops_optimizer.py
```

## **Contribuição**

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## **Estrutura do Projeto**

```
aws-finops-optimizer/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── ecs_finops_optimizer.py
```

## **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.