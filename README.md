</li><li><strong>Libraries installation command:</strong>

```
pip install langchain langgraph langchain-google-genai langchain_community tavily-python python-dotenv aiosqlite
```

<br>

</li><li><strong>Banco de Dados para Checkpoints:</strong> O projeto usará <code>sqlite</code> para persistir o estado do grafo, permitindo pausas e retomadas. Nenhuma configuração adicional é necessária além da instalação da biblioteca, pois o banco de dados será criado quando o código for executado pela primeira vez.</li></ul>