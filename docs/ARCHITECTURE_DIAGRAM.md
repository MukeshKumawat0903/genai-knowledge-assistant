# LLM Abstraction Layer - Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consumer Code (RAG, Agents, UI)              │
│                                                                  │
│  from app.core.llm import LLMFactory                           │
│  llm = LLMFactory.create()  # ← Single line to get configured LLM│
│  response = llm.invoke("What is RAG?")                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                       LLMFactory                                 │
│                    (Factory Pattern)                             │
│                                                                  │
│  @staticmethod                                                  │
│  def create() -> Any:                                           │
│      settings = get_settings()                                  │
│      provider = settings.llm_provider                           │
│                                                                  │
│      if provider == "groq":                                     │
│          return GroqLLM().get_llm()                            │
│      # ... more providers                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                         BaseLLM                                  │
│                   (Abstract Base Class)                          │
│                                                                  │
│  class BaseLLM(ABC):                                            │
│      @abstractmethod                                            │
│      def get_llm(self) -> Any:                                 │
│          """Returns LangChain chat model"""                    │
│          pass                                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┬─────────────────┐
         ↓                               ↓                 ↓
┌─────────────────┐                       ┌──────────────────┐
│   GroqLLM       │                       │ [Future Providers]│
│   (Cloud)       │                       │                  │
├─────────────────┤                       │ • OpenAILLM      │
│ get_llm():      │                       │ • AnthropicLLM   │
│   settings = ✓  │                       │ • GoogleLLM      │
│   validate_key  │                       │ • AzureOpenAI    │
│   return        │                       └──────────────────┘
│   ChatGroq()    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ChatGroq       │
│  (LangChain)    │
├─────────────────┤
│ • invoke()      │
│ • stream()      │
│ • batch()       │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Groq Cloud API │
│  (https://...)  │
└─────────────────┘
```

## Configuration Flow

```
┌──────────────┐
│  .env file   │
│              │
│ LLM_PROVIDER │
│ LLM_MODEL    │
│ API_KEYS     │
│ TEMPERATURE  │
│ MAX_TOKENS   │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│   Settings       │  ← Singleton pattern (loaded once)
│   (config.py)    │
│                  │
│ • Loads .env     │
│ • Validates      │
│ • Provides       │
│   get_settings() │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  LLMFactory      │  ← Reads from Settings
│  BaseLLM classes │  ← Access via get_settings()
└──────────────────┘
```

## Class Relationships (UML-style)

```
                    ┌─────────────────┐
                    │     BaseLLM     │
                    │   <<abstract>>  │
                    ├─────────────────┤
                    │ + get_llm()     │
                    └────────△────────┘
                             │
                   implements│
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────┴────────┐ ┌───┴────────┐ ┌──┴─────────┐
           │    GroqLLM      │ │  Future... │
           ├─────────────────┤ └────────────┘
           │ + get_llm()     │
           │   → ChatGroq    │
           └─────────────────┘
            △
            │
            └──────┬───────┘
                     │ creates
              ┌──────┴────────┐
              │  LLMFactory   │
              ├───────────────┤
              │ + create()    │
              │   → BaseLLM   │
              └───────────────┘
                     △
                     │ uses
              ┌──────┴────────┐
              │ Consumer Code │
              │  (RAG/Agent)  │
              └───────────────┘
```

## Data Flow - Single Request

```
1. User Request
   │
   ↓
2. Consumer Code
   │  llm = LLMFactory.create()
   ↓
3. LLMFactory
   │  settings = get_settings()
   │  provider = settings.llm_provider
   ↓
4. Settings (Singleton)
   │  Returns: "groq"
   ↓
5. LLMFactory
   │  return GroqLLM().get_llm()
   ↓
6. GroqLLM
   │  settings = get_settings()
   │  Validates: API key exists
   │  Creates: ChatGroq instance
   ↓
7. ChatGroq (LangChain)
   │  Ready for inference
   ↓
8. Consumer Code
   │  response = llm.invoke("What is RAG?")
   ↓
9. ChatGroq → Groq API
   │  HTTP request with prompt
   ↓
10. Groq API Response
    │  Generated text
    ↓
11. ChatGroq
    │  Parses response
    ↓
12. Consumer Code
    │  Receives: AIMessage object
    │  Displays: response.content
```

## Design Patterns Applied

```
┌──────────────────────────────────────────────────────────┐
│                    Factory Pattern                        │
├──────────────────────────────────────────────────────────┤
│ Problem:  Need to create different LLM objects          │
│ Solution: LLMFactory.create() encapsulates creation     │
│ Benefit:  Consumer doesn't know concrete classes        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              Abstract Base Class Pattern                  │
├──────────────────────────────────────────────────────────┤
│ Problem:  Need consistent interface across providers     │
│ Solution: BaseLLM defines contract with get_llm()       │
│ Benefit:  All implementations are interchangeable        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  Singleton Pattern                        │
├──────────────────────────────────────────────────────────┤
│ Problem:  Don't want to reload .env multiple times      │
│ Solution: get_settings() caches Settings instance       │
│ Benefit:  Efficient, consistent configuration            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              Strategy Pattern (implicit)                  │
├──────────────────────────────────────────────────────────┤
│ Problem:  Different algorithms for different providers   │
│ Solution: Each LLM class encapsulates its strategy      │
│ Benefit:  Easy to add/modify provider logic              │
└──────────────────────────────────────────────────────────┘
```

## SOLID Principles Demonstrated

```
┌─────────────────────────────────────────────────────────┐
│ S - Single Responsibility Principle                     │
├─────────────────────────────────────────────────────────┤
│ ✓ GroqLLM:      Only handles Groq initialization       │
│ ✓ LLMFactory:   Only handles LLM creation              │
│ ✓ Settings:     Only handles configuration             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ O - Open/Closed Principle                               │
├─────────────────────────────────────────────────────────┤
│ ✓ Open for extension:   Add new provider classes       │
│ ✓ Closed for modification: Don't change BaseLLM        │
│   Example: Add OpenAILLM without touching existing code │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ L - Liskov Substitution Principle                       │
├─────────────────────────────────────────────────────────┤
│ ✓ Any BaseLLM subclass can replace another             │
│   Example: Switch providers, code still works           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ I - Interface Segregation Principle                     │
├─────────────────────────────────────────────────────────┤
│ ✓ BaseLLM has minimal interface (only get_llm())       │
│ ✓ No client forced to depend on unused methods         │
│ ✓ Clean, focused abstraction                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ D - Dependency Inversion Principle                      │
├─────────────────────────────────────────────────────────┤
│ ✓ High-level code depends on BaseLLM (abstraction)     │
│ ✓ Not on GroqLLM (concrete implementation)             │
│ ✓ Both depend on abstraction, not on each other        │
└─────────────────────────────────────────────────────────┘
```

## Extensibility Example

Adding a new provider requires:

```
Step 1: Create new class (app/core/llm.py)
┌─────────────────────────────────────┐
│ class OpenAILLM(BaseLLM):           │
│     def get_llm(self):              │
│         # Implementation            │
│         return ChatOpenAI(...)      │
└─────────────────────────────────────┘

Step 2: Update factory (app/core/llm.py)
┌─────────────────────────────────────┐
│ def create():                       │
│     # ... existing code             │
│     elif provider == "openai":      │
│         return OpenAILLM().get_llm()│
└─────────────────────────────────────┘

Step 3: Update settings (app/utils/config.py)
┌─────────────────────────────────────┐
│ llm_provider: Literal[              │
│     "openai",  # ← Add here         │
│     "groq",                          │
│ ]                                    │
└─────────────────────────────────────┘

✅ Done! No changes needed in consumer code
```

## Provider Comparison Matrix

```
┌────────────┬──────────┬──────────┬──────────┬──────────────┐
│ Provider   │ Location │ Speed    │ Privacy  │ Cost         │
├────────────┼──────────┼──────────┼──────────┼──────────────┤
│ Groq       │ Cloud    │ ⚡⚡⚡     │ Medium   │ Free tier    │
│ OpenAI     │ Cloud    │ ⚡⚡      │ Medium   │ Pay-per-use  │
│ Anthropic  │ Cloud    │ ⚡⚡      │ Medium   │ Pay-per-use  │
└────────────┴──────────┴──────────┴──────────┴──────────────┘

⚡ = Fast     ⭐ = High
```

## Interview-Ready Architecture Explanation

**Interviewer**: "Walk me through your LLM abstraction layer."

**You**:

1. **Purpose**: "I designed an abstraction layer to prevent vendor lock-in and make it trivial to switch between LLM providers."

2. **Architecture**: "It uses three key components:
   - `BaseLLM`: Abstract base class defining the contract
   - Concrete implementations: `GroqLLM` (and future providers)
   - `LLMFactory`: Factory for centralized creation"

3. **Design Patterns**:
   - "Factory pattern for object creation
   - Abstract base class for interface consistency
   - Singleton for configuration management"

4. **SOLID Principles**: "Each class has one responsibility, open for extension but closed for modification, and we depend on abstractions not concrete classes."

5. **Real Benefits**:
   - "Switch providers by changing one env variable
   - Add new providers without touching consumer code
   - Consistent interface across all LLMs
   - Easy to test with mocks"

6. **Production-Ready**: "All configuration externalized, proper error handling, type hints throughout, comprehensive documentation."

---

This architecture demonstrates professional software engineering practices suitable for production systems and technical interviews.
