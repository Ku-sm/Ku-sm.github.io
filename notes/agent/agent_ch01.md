# Agent

## LangChain

For LLMs, two key challenges arise:
1. Adding short- and long-term memory to models
2. Handling different situations that require different models, since there are no universal rules

**LangChain** is a framework that addresses these challenges — it enables application development on top of language models.
- Supports data summarization, analysis, and generation over documents or structured data, as well as code understanding and API interaction
- Available in both Python and TypeScript/JavaScript

## How LangChain Works

LangChain is organized around **6 modules**:

| Module | Description |
|---|---|
| Model I/O | Defines the interface with language models |
| Data Connection | Defines the interface for application-specific data |
| Chains | Sequences of calls |
| Agents | Tool selection driven by instructions |
| Memory | Maintains state and information across chain calls |
| Callbacks | Logs and streams intermediate steps between chains |

## Advantages of LangChain

1. Standard Model Interface
2. Easy to Use
3. LangGraph
4. Debug with LangSmith

### 1. Standard Model Interface

- **Tool calling**: Access external APIs or databases
- **Structured output**: Return output in a defined schema
- **Multimodality**: Supports text, image, audio, and video
- **Reasoning**: Easy to leverage model reasoning capabilities

The model can be used standalone, or configured as an agent through various compositions.

Three core invocation methods:

| Method | Description |
|---|---|
| `invoke` | Query the model with one or more messages |
| `stream` | Receive output from the model token by token |
| `batch` | Process multiple queries in parallel |

```python
# Invoke — multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
```

```python
# Stream — display output incrementally
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]
```

```python
# Batch — parallel processing of multiple queries
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in responses:
    print(response)
```

### 2. Easy to Use

#### Model

The reasoning engine of an agent. Can be selected statically or dynamically.

- **Static model**: configured once at creation, unchanged throughout execution
- **Dynamic model**: selected at runtime based on current state and context, enabling routing logic and cost optimization

```python
# Static model
from langchain.agents import create_agent
agent = create_agent("openai:gpt-5.4", tools=tools)
```

```python
# Dynamic model
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-5.4-mini")
advanced_model = ChatOpenAI(model="gpt-5.4")

# Switch to a more capable model after 10 messages
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])
    model = advanced_model if message_count > 10 else basic_model
    return handler(request.override(model=model))

# Model switching is handled via middleware
agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

#### Tools

Tools give agents the ability to take actions. They enable:
- Multiple tool calls in sequence
- Parallel tool calls when appropriate
- Dynamic tool selection based on previous results
- Tool retry logic and error handling
- State persistence across tool calls

**1. Static tools** — tools that do not change during execution:

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Declare a function as a tool using the @tool decorator
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Pass tools at agent creation; without tools, the agent acts as a plain LLM
agent = create_agent(model, tools=[search, get_weather])
```

**2. Dynamic tools** — overridable per agent; suited for permission state, user authorization, feature flags, and conversation state.

Via middleware (preferred approach):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

# Filter available tools based on authentication state
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation state."""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-5.4",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

Via tool access (alternative approach):

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Inject a dynamic tool (e.g., loaded from an MCP server or database)
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],  # only static tools registered here
    middleware=[DynamicToolMiddleware()],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate a 20% tip on $85"}]
})
```

#### System Prompt

The system prompt shapes how the agent approaches problems.

```python
# Set a static system prompt at creation
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

# System prompts can also be passed as a SystemMessage object
SystemMessage(
    content=[
        {
            "type": "text",
            "text": "You are an AI assistant tasked with analyzing literary works.",
        },
        {
            "type": "text",
            "text": "<the entire contents of 'Pride and Prejudice'>",
            "cache_control": {"type": "ephemeral"}
        }
    ]
)

# System prompts can also be dynamic
class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-5.4",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)
```

#### Name

Agents can be given a name, which serves as an identifier in multi-agent systems.

```python
agent = create_agent(
    model,
    tools,
    name="research_assistant"
)
```

### Invocation

Since the agent maintains internal state, use `invoke` to query it:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

### Memory

By default, agents automatically maintain conversation history through message state. A custom state schema can be used to persist additional information.

Two custom state schema types: `AgentState` and `TypedDict`.
Two definition methods: **middleware** (recommended) and the **`state_schema`** argument at creation.

#### Defining state via middleware

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

#### Defining state via `state_schema`

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

### Streaming

While `invoke` waits for the final response, streaming shows intermediate steps as the agent progresses through multiple stages.

```python
from langchain.messages import AIMessage, HumanMessage

messages = [{"role": "user", "content": "Search for AI news and summarize the findings"}]
for chunk in agent.stream({"messages": messages}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

### Middleware

Middleware provides powerful extension points for customizing agent behavior at different execution stages:

- Process state before the model is called
- Modify or validate the model's response
- Handle tool execution errors with custom logic
- Implement dynamic model selection based on state or context
- Add custom logging, monitoring, or analytics

---

## References

[1] [Samsung SDS](https://www.samsungsds.com/kr/insights/the-concept-of-langchain.html)
[2] [LangChain Docs](https://docs.langchain.com/oss/python/langchain/overview)
