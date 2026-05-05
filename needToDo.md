## what to do next?
* build a code graph (AST-based indexing)

## What to learn about and clear my concepts?
* llm parameters?
* how vectorsearch is working?
* how model is thinking in token? (Four people must cross a rickety bridge at night with only one flashlight. The bridge can only hold two people at a time. Person A takes 1 min, B takes 2, C takes 5, and D takes 10 mins. If they cross together, they move at the speed of the slower person. How can they all cross in 17 minutes?)

* Yes you can use it — but the problem is qwen2.5-coder:7b doesn't support function calling properly through LangGraph's create_react_agent. It outputs tool calls as plain JSON text instead of actually executing them.


* how lagngraph handle tool execution?

`LangGraph's create_react_agent is a black box.
Writing the loop yourself means you UNDERSTAND
exactly what an agent loop is doing step by step.
That's the whole point of this learning project.`
