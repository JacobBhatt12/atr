"""A tiny todo API, the example app for `vrtx deploy ./examples/todo-api`."""
import os
import socket

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="todo-api")
todos: dict[int, dict] = {}


class Todo(BaseModel):
    title: str
    done: bool = False


@app.get("/")
def index():
    # which replica answered: handy for watching the gateway spread requests
    return {"app": "todo-api", "replica": os.getenv("VRTX_REPLICA", socket.gethostname()),
            "version": os.getenv("VRTX_VERSION"), "todos": len(todos)}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/todos")
def list_todos():
    return list(todos.values())


@app.post("/todos", status_code=201)
def add(todo: Todo):
    tid = max(todos, default=0) + 1
    todos[tid] = {"id": tid, **todo.model_dump()}
    return todos[tid]


@app.delete("/todos/{tid}")
def remove(tid: int):
    if tid not in todos:
        raise HTTPException(404, "no such todo")
    return todos.pop(tid)
