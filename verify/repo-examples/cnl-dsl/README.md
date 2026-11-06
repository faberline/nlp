# Repository examples for CNL to DSL

These cases use a small part of an upstream repository to exercise SOM's
compiler contract. Each case states what it models and what it leaves out.

## Defer queue and task

The [Defer task contract](https://github.com/faberline/defer/blob/5f7adac1fe1333b33e3ec69a92b5ca4a634a97d0/docs/product/tasks-and-queues.md#scheduled-tasks)
places a task under a queue. This case models the relationship, a few task
fields, and the write in one database transaction. SOM uses its own integer
queue ID. The case leaves out dispatch, retry, and retention.

The paired typed prompt records the expected input to the planner. The test
also checks that the DSL selects a PostgreSQL transaction and preserves the
queue reference. A second test uses a string input for the queue ID. NLP
accepts the syntax and SOM rejects the reference at its use.
