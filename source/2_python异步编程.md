---
title:
date: "2026-11-01"
tags:
comment: true
toc: true
published: true
permalink:
hexo-path:
---
#
<!--more-->

# 1. 简介

在IO密集型任务（如网络请求、数据库操作）中，传统的同步代码会阻塞等待，导致CPU空闲。`asyncio`提供基于**事件循环**的**单线程并发方案**：遇到**IO等待时**自动切换任务，从而大幅提升吞吐量。尤其适合AI服务中需要同时调用多个外部API的场景。
> 例如有两个程序，A先sleep2秒，然后输出hello。 B先sleep2秒然后输出world。如果采用同步编程，那么两个函数跑完需要4秒。异步编程就是先发现两个函数的sleep操作是可以并行的，当A在sleep的时候可以将执行权让出去给B执行，而不是让整个任务进度卡在A这里。让出执行权这个操作就可以提高吞吐量，让出来之后B马上也进入sleep，这样两个函数的sleep时间就重叠了。
# 2. 语法
## 2.1 定义协程
- 协程函数在调用时不会立即执行，而是返回一个协程对象
- 通常协程里面会有异步操作，这个操作就是可以优化的点。异步操作通常用await来调用启动。
```python
import asyncio

# 1. 定义协程
async def say_hello():
    # 先输出hello，一秒后再输出world
    print('hello')
    await asyncio.sleep(1)
    print('world')

# 2. 运行协程
if __name__ == '__main__':
    asyncio.run(say_hello())
```
## 2.2 await关键字
- 挂起当前协程，等待另一个可等待对象（协程、Task、Future）完成。
- 运行到await的时候代码让出执行权，事件循环队列执行下一个事件，当前代码相当于注册了一个过了多久唤醒我的预约操作。
```python
import asyncio

async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

async def main():
    print("开始")
    await say_hello()      # 等待say_hello执行完毕
    print("结束")

asyncio.run(main())
```
## 2.3 创建任务
- 用`asyncio.create_task()`将协程包装为任务，使其并发执行。
```python
import asyncio

async def say_after(delay, msg):
    await asyncio.sleep(delay)
    print(msg)

async def main():
    # 创建两个任务，它们将并发执行
    task1 = asyncio.create_task(say_after(3, "Hello"))
    task2 = asyncio.create_task(say_after(1, "World"))
    
    print("任务已创建，等待完成...")
    await task1   # 等待task1完成
    await task2   # 等待task2完成
    print("所有任务完成")

asyncio.run(main())
```
代码流程：

1. 启动事件循环：`asyncio.run(main())` 启动事件循环，就绪队列里面现在只有入口函数【main】，然后执行 `main()`。
2. 执行main：
    - `create_task()` 创建并调度 `task1` 和 `task2`，放入就绪队列。【task1, task2】（main正在执行，不在就绪队列中）
    - 打印
    - 执行 `await task1`，由于task1还未完成，当前函数`main()` 暂停，让出执行权。
3. 事件循环调度任务，由于createtask先创建了task1，因此执行权让给task1。
4. 事件循环执行 `task1`
    - 执行`say_after`函数。
    - 遇到 `await asyncio.sleep(3)`，暂停并让出执行权。同时自己进入等待状态，不在就绪队列中。（这里要注意，让出执行权之后sleep等待是可以在后台继续计时的，因为系统时钟一直在前进，事件循环根据时钟判断等待是否结束。它只需要记录一个唤醒时间。）
5. 事件循环执行 `task2`，遇到 `await asyncio.sleep(1)`，暂停并让出执行权。
6. 此时main函数在等待。task1的sleep时间还没到，因此也在等待。task2也进入等待。就绪队列为空，事件循环等待定时器，CPU可以空闲。
7. 约 1 秒后，`task2` 的sleep到期，task2进入就绪队列，事件循环恢复task2。
    - 输出World，任务结束。让出执行权，此时main和task1仍在等待状态，就绪队列为空。
8. main还在await task1，task1的sleep还没到期，事件循环继续等待，就绪队列为空。
9. 约 3 秒后，`task1` 由于有注册时间，因此进入就绪队列并恢复执行，输出Hello，任务结束。此时main等待的条件满足，main进入就绪队列【main】。
10. main函数的await task1恢复，往后执行await task2。
11. task2在第7步的时候就运行完了，因此`await task2`直接通过，输出“所有任务完成”，main结束，事件循环关闭。
总耗时约 3 秒。核心：`create_task` 负责调度并发，`await` 在需要等待时暂停当前协程，事件循环负责切换任务。

# 3. 多协程并发
## 3.1 `asyncio.gather()`——收集结果
- 让多个任务并发执行，并等待全部完成，按传入参数的顺序组织结果。跟2.3的例子本质上是一样的，只是任务管理方式的差别。
```python
import asyncio

async def say_after(delay, msg):
    await asyncio.sleep(delay)
    print(msg)

async def main():
    print("任务开始")

    await asyncio.gather(
        say_after(3, "Hello"),
        say_after(1, "World")
    )

    print("所有任务完成")

asyncio.run(main())
```
## 3.2 `asyncio.wait()`——细粒度控制
- 可以设置超时、等待条件（FIRST_COMPLETED、ALL_COMPLETED、FIRST_EXCEPTION），从而只需要等待部分任务完成。返回完成和未完成的任务集合。
```python
import asyncio

async def fetch_data(delay, name):
    await asyncio.sleep(delay)
    return f"{name} 完成"

async def main():
    tasks = {asyncio.create_task(fetch_data(i, f"任务{i}")) for i in [3, 1, 2]}
    # 等待第一个任务完成，超时2秒
    done, pending = await asyncio.wait(tasks, timeout=2, return_when=asyncio.FIRST_COMPLETED)
    
    print(f"已完成的任务数: {len(done)}")
    for task in done:
        print(task.result())
    
    # 取消未完成的任务
    for task in pending:
        task.cancel()
        print("已取消一个任务")

asyncio.run(main())
```
## 3.3 `asyncio.as_completed()` —— 按完成顺序处理
- 返回一个迭代器，按任务完成的先后顺序产出结果。

```python
import asyncio

async def fetch_data(delay, name):
    await asyncio.sleep(delay)
    return f"{name} 完成"

async def main():
    tasks = [fetch_data(i, f"任务{i}") for i in [3, 1, 2]]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)   # 按实际完成顺序打印：任务1完成、任务2完成、任务3完成

asyncio.run(main())
```
# 4. 可等待对象
- 可等待对象包括：协程、Task、Future。通常我们只需处理协程和Task。
# 5. 异步上下文管理器
- 用于管理需要异步初始化和清理的资源。例如aiohttp的ClientSession、异步数据库连接等。
- 自定义异步上下文管理器需实现`__aenter__`和`__aexit__`。
```python
import asyncio
import asyncpg

class AsyncDB:
    async def __aenter__(self):
        self.conn = await asyncpg.connect("postgresql://postgres:123456@localhost:5432/postgres")
        return self.conn

    async def __aexit__(self, *_):
        await self.conn.close()

async def query1():
    async with AsyncDB() as conn:
        await conn.execute("SELECT pg_sleep(2)")
        return await conn.fetchval("SELECT current_database()")

async def query2():
    async with AsyncDB() as conn:
        await conn.execute("SELECT pg_sleep(1)")
        return await conn.fetchval("SELECT COUNT(*) FROM pg_tables")

async def main():
    results = await asyncio.gather(query1(), query2())
    print(results)

asyncio.run(main())
```
# 6. 异步迭代器
1. 先看同步的情况：有两个任务，代码总耗时6秒
```python
import time

def generate(name):
    for i in range(3):
        time.sleep(1)
        yield f"{name}-{i}"

start = time.time()

for x in generate("A"):
    print(x)

for x in generate("B"):
    print(x)

print("耗时:", time.time() - start)
```
2. 再看异步的情况：代码总耗时3秒。当有多个任务时，多个任务的等待部分就能重叠，整体速度就能提升了。
```python
import asyncio
import time

async def generate(name):
    for i in range(3):
        await asyncio.sleep(1)
        yield f"{name}-{i}"

async def consume(name):
    async for x in generate(name):
        print(x)

async def main():
    await asyncio.gather(
        consume("A"),
        consume("B")
    )

start = time.time()
asyncio.run(main())
print("耗时:", time.time() - start)
```
代码执行流程
1. 启动事件循环：就绪队列里面只有【main】，然后执行 `main()`。
2. 执行 main：
    - 包装两个 Task，放入就绪队列【consume_A, consume_B】。
    - 执行 `await asyncio.gather()`，由于两个任务还未完成，`main()` 暂停，让出执行权。
3. 事件循环执行 consume_A：
    - 遇到 `async for x in generate("A")`，调用异步生成器 `generate("A")`。
    - `async for` 请求第一个元素，开始执行 `generate("A")`。
    - 遇到 `await asyncio.sleep(1)`，`consume_A` 暂停并让出执行权，等待1秒。
    - 此时就绪队列【consume_B】。
4. 事件循环执行 consume_B：
    - 遇到 `async for x in generate("B")`，调用异步生成器 `generate("B")`。
    - `async for` 请求第一个元素。
    - 遇到 `await asyncio.sleep(1)`，`consume_B` 暂停并让出执行权，等待1秒。
5. 此时三个任务全部等待：
    - `main` 等待 `gather` 完成。
    - `consume_A` 等待第一个元素的1秒定时器。
    - `consume_B` 等待第一个元素的1秒定时器。
    - 就绪队列为空，事件循环等待定时器。
6. **约1秒后**，两个 sleep 到期：
    - `consume_A` 和 `consume_B` 先后进入就绪队列。
    - 事件循环恢复 `consume_A`，执行 `yield "A-0"`，将数据返回给 `async for`。
    - `consume_A` 执行 `print(x)`，输出 `A-0`。
    - `async for` 请求第二个元素，生成器继续执行，遇到下一次 `await asyncio.sleep(1)`，再次暂停。让出执行权，由于刚刚AB是前后脚进入队列的，B可以恢复了。
    - 事件循环恢复 `consume_B`，同样执行 `yield "B-0"`，输出 `B-0`，然后进入下一次异步等待。
7. 此时两个任务再次等待：
    - `consume_A` 等待第二个元素。
    - `consume_B` 等待第二个元素。
    - `main` 仍在等待 `gather`。
    - 就绪队列为空。
8. 约2秒后：
    - 两个任务的第二次 sleep 到期。
    - `consume_A` 恢复，执行 `yield "A-1"`，输出 `A-1`。
    - `consume_B` 恢复，执行 `yield "B-1"`，输出 `B-1`。
    - 两个生成器再次执行 `await asyncio.sleep(1)`，暂停并让出执行权。
9. 约3秒后：
    - 两个任务的第三次 sleep 到期。
    - `consume_A` 恢复，执行 `yield "A-2"`，输出 `A-2`。
    - `consume_B` 恢复，执行 `yield "B-2"`，输出 `B-2`。
    - 两个生成器下一次迭代时都结束，`async for` 停止遍历。
    - `consume_A` 和 `consume_B` 全部完成。
10. 恢复 main
11. 执行最后的同步代码，总耗时约3秒。

# 7. 同步到异步
- 上面的例子中，我们需要定义协程函数，然后里面有可重叠的异步操作。有的时候一个同步函数里面也有这种可重叠的操作，如何把一个同步函数也做异步处理呢？
```python
import asyncio
import time

def blocking_io():
    """模拟阻塞IO操作（如文件读写、同步请求）"""
    time.sleep(2)
    return "阻塞操作结果"

async def main():
    loop = asyncio.get_running_loop() #获取当前事件循环。
    # 在线程池中运行阻塞函数，通过子线程来运行
    result = await loop.run_in_executor(None, blocking_io)
    print(result)

asyncio.run(main())
```
这里执行run_in_executor的时候函数就已经提交给子线程来执行了。

# 8. 异步队列
- 用于在多个协程之间安全传递数据，典型的生产者-消费者模型。
```python
import asyncio
import random

async def producer(queue, n):
    for i in range(n):
        item = f"物品{i}"
        await queue.put(item)
        print(f"生产了 {item}")
        await asyncio.sleep(random.random())
    await queue.put(None)   # 发送结束信号

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"消费了 {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=5)
    prod_task = asyncio.create_task(producer(queue, 10))
    cons_task = asyncio.create_task(consumer(queue))
    await asyncio.gather(prod_task, cons_task)

asyncio.run(main())

```
代码执行流程
1. 启动事件循环
	- `asyncio.run(main())` 启动事件循环，执行 `main()`。
	- 当前就绪队列只有【main】。
2. 执行 main
	- `queue = asyncio.Queue(maxsize=5)` 创建容量为 5 的异步队列。
	- `create_task()` 创建并调度 `producer` 和 `consumer`。
	- 此时就绪队列为【producer, consumer】。
	- 执行 `await asyncio.gather(...)`，`main` 暂停，让出执行权。
3. 事件循环执行 producer
	- 进入 `for i in range(10)`。
	- 创建 `"物品0"`。
	- 执行 `await queue.put("物品0")`。
	- 因为队列未满，立即放入队列，不需要等待。
	- 输出 `生产了 物品0`。
	- 执行 `await asyncio.sleep(random.random())`，暂停并让出执行权。
4. 事件循环执行 consumer
	- 进入 `while True`。
	- 执行 `await queue.get()`，从队列取出 `"物品0"`。
	- 因为队列非空，立即返回，不需要等待。
	- 输出 `消费了 物品0`。
	- 执行 `queue.task_done()`，标记当前物品处理完成。
	- 再次执行 `await queue.get()`，由于队列为空，`consumer` 暂停并让出执行权。
5. 此时两个任务都在等待
	- `producer` 等待 `sleep` 定时器到期。
	- `consumer` 等待队列中出现新数据。
	- `main` 等待两个任务完成。
	- 就绪队列为空，事件循环等待新事件。
6. producer 的 sleep 到期
	- `producer` 进入就绪队列并恢复执行。
	- 创建 `"物品1"`。
	- 执行 `await queue.put("物品1")`，将数据放入队列。
	- 队列通知等待中的 `consumer`，使其可以恢复执行。
	- `producer` 输出 `生产了 物品1`。
	- 再次遇到 `await asyncio.sleep(...)`，暂停并让出执行权。
7. consumer 恢复
	- `await queue.get()` 返回 `"物品1"`。
	- 输出 `消费了 物品1`。
	- 执行 `queue.task_done()`。
	- 再次等待下一个物品。
8. 重复生产和消费
# 9. 超时与任务取消
## 9.1 超时
- 设置协程的最大执行时间，超时抛出`TimeoutError`。
- 这跟3.2的区别是啥？`wait_for()` 是带超时取消机制的等待；`wait()` 是返回任务完成状态、由你自行管理剩余任务的等待。
```python
import asyncio

async def long_operation():
    await asyncio.sleep(10)
    return "完成"

async def main():
    try:
        result = await asyncio.wait_for(long_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("操作超时")

asyncio.run(main())
```

## 9.2 取消任务

```python
import asyncio

async def cancellable_task():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("任务被取消，进行清理...")
        raise   # 必须重新抛出以标记任务取消

async def main():
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(1) #执行到这里会启动task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("任务已取消")

asyncio.run(main())
```
代码执行流程
1. `asyncio.run(main())` 启动事件循环，执行 `main()`。
2. `create_task(cancellable_task())` 创建并调度任务，但此时任务通常还未执行。
3. `main` 遇到 `await asyncio.sleep(1)`，暂停并让出执行权。
4. 事件循环执行 `cancellable_task()`，遇到 `await asyncio.sleep(10)`，任务暂停。
5. 约 1 秒后，事件循环恢复 `main`，执行 `task.cancel()`，向任务发送取消请求。
6. 任务在 `await asyncio.sleep(10)` 处收到 `CancelledError`。
7. 执行异常处理，打印 `任务被取消，进行清理...`，随后 `raise` 重新抛出异常。
8. `main` 的 `await task` 收到 `CancelledError`，打印 `任务已取消`。
## 9.3 屏蔽取消
- 保护某个协程不被取消，即使外层任务被取消。
```python
import asyncio

async def child():
    try:
        await asyncio.sleep(3)
        print("子任务完成")
    except asyncio.CancelledError:
        print("子任务被取消")
        raise

async def main(task):
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        print("main收到取消异常")
        try:
            await task
        except asyncio.CancelledError:
            print("确认子任务已取消")

async def test(cancel_child):
    task = asyncio.create_task(child())
    main_task = asyncio.create_task(main(task))

    await asyncio.sleep(1)

    if cancel_child:
        task.cancel()       # 取消子任务
    else:
        main_task.cancel()  # 取消主任务

    await main_task

async def run():
    print("=== 测试1：取消主任务 ===")
    await test(False)

    print("=== 测试2：取消子任务 ===")
    await test(True)

asyncio.run(run())

```

# 10. 限制并发数量（信号量）

- 使用`asyncio.Semaphore`控制同时执行的任务数量，例如限制对API的并发请求。
```python
import asyncio
import aiohttp   # 需要安装：pip install aiohttp

async def fetch_url(sem, session, url):
    async with sem: # 这一步也有await
        print(f"正在请求: {url}")
        async with session.get(url) as response: #这一步也有await
            return await response.text() # 异步方法需要await

async def main():
    urls = ["http://example.com"] * 10   # 模拟10个相同请求
    sem = asyncio.Semaphore(3)   # 最多同时3个请求
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(sem, session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"获取到 {len(results)} 个响应")
        
asyncio.run(main())
```
代码执行流程
1. 启动事件循环
	- 启动事件循环，就绪队列里面只有【main】，然后执行 `main()`。
2. 执行 main
	- 创建 `urls`，包含 10 个相同的 URL。
	- `asyncio.Semaphore(3)` 创建信号量，最多允许 3 个协程同时进入受保护的代码区域。
	- `async with aiohttp.ClientSession()` 创建 HTTP 客户端会话，负责管理网络连接。
	- `tasks = [...]` 创建 10 个协程对象，此时尚未执行。
	- 执行 `await asyncio.gather(*tasks)`，将 10 个协程包装成 Task 并调度。
	- `main` 暂停，让出执行权。
3. 事件循环执行前 3 个任务
	- `task1` 执行 `async with sem`，获得第 1 个信号量许可。
	- `task2` 获得第 2 个许可。
	- `task3` 获得第 3 个许可。
	- 三个任务分别执行 `session.get(url)`，发送 HTTP 请求。
	- 遇到网络 I/O 等待时，各自暂停，让出执行权。
	- 此时最多有 3 个任务正在请求或读取响应。
4. 事件循环执行 task4～task10
	- 这些任务也执行 `async with sem`。
	- 但此时 3 个许可已经全部被占用。
	- 因此剩余 7 个任务进入等待状态，不会立即发送 HTTP 请求。
	- `main` 仍然等待 `gather()` 完成。
5. 假设约 1 秒后，task2 请求完成
	- 事件循环恢复 `task2`。
	- 执行 `await response.text()`，如果响应内容已经准备好，就返回 HTML。
	- 退出 `async with session.get()`，释放响应相关资源。
	- 退出 `async with sem`，释放 1 个信号量许可。
	- `task2` 结束。
	- 此时 `task4` 可以获得许可，开始发送请求。
6. 所有任务执行完成
# 11. 应用场景
## 11.1 并发调用API
假设需要同时对文本进行三种不同的AI处理。
```python
import asyncio
import aiohttp

async def call_ai_api(session, text, model_name):
    """模拟调用AI API"""
    url = f"https://api.example.com/{model_name}/predict"
    payload = {"text": text}
    async with session.post(url, json=payload) as resp:
        return await resp.json()

async def handle_request(text):
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_ai_api(session, text, "sentiment"),
            call_ai_api(session, text, "ner"),
            call_ai_api(session, text, "summary")
        ]
        results = await asyncio.gather(*tasks)
        return {"text": text, "sentiment": results[0], "ner": results[1], "summary": results[2]}

async def main():
    result = await handle_request("I love Python async programming!")
    print(result)

# 实际运行时需要真实API端点，此处仅展示结构
asyncio.run(main())
```
## 11.2 异步数据库访问
```python
import asyncio
import asyncpg

async def fetch_users():
    conn = await asyncpg.connect(
        user='your_user', password='your_pass',
        database='your_db', host='localhost'
    )
    rows = await conn.fetch('SELECT id, name FROM users LIMIT 10')
    await conn.close()
    return rows

async def main():
    users = await fetch_users()
    for user in users:
        print(dict(user))

# 注意：需替换为真实数据库连接信息
# asyncio.run(main())
```
## 11.3 异步微服务网关
同时向后端多个服务发起请求，合并响应。
```python
import asyncio
import aiohttp

async def fetch_service(session, service_url, payload):
    async with session.post(service_url, json=payload) as resp:
        return await resp.json()

async def gateway(request_data):
    services = [
        "http://ai-service1/predict",
        "http://ai-service2/predict",
        "http://ai-service3/predict"
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_service(session, url, request_data) for url in services]
        results = await asyncio.gather(*tasks)
        return results

async def main():
    data = {"input": "test data"}
    combined = await gateway(data)
    print(combined)

asyncio.run(main())
```
## 11.4 流式数据处理（异步迭代器 + 队列）
从消息队列消费消息，并发处理。
```python
import asyncio
import random

async def message_generator(queue):
    """模拟消息生产者"""
    for i in range(20):
        await asyncio.sleep(random.random() * 0.5)
        msg = f"消息{i}"
        await queue.put(msg)
        print(f"放入: {msg}")
    await queue.put(None)   # 结束信号

async def process_message(msg):
    """模拟AI处理"""
    await asyncio.sleep(random.random())
    return f"处理结果: {msg}"

async def worker(worker_id, queue, results):
    while True:
        msg = await queue.get()
        if msg is None:
            await queue.put(None)   # 传递结束信号给其他worker
            break
        result = await process_message(msg)
        print(f"Worker {worker_id} 处理了 {msg} -> {result}")
        results.append(result)

async def main():
    queue = asyncio.Queue()
    results = []
    
    # 启动生产者和消费者
    producer = asyncio.create_task(message_generator(queue))
    workers = [asyncio.create_task(worker(i, queue, results)) for i in range(3)]
    
    await producer
    await asyncio.gather(*workers)
    print(f"总共处理了 {len(results)} 条消息")

asyncio.run(main())
```
# 12. 常见错误与调试技巧
## 12.1 忘记await
```python
import asyncio

async def bad_example():
    # 错误：直接调用协程但没有await，协程不会执行
    asyncio.sleep(1)   # 缺少await，返回一个协程对象，未执行
    print("这行会立即打印")

async def good_example():
    await asyncio.sleep(1)
    print("1秒后打印")

asyncio.run(good_example())

```
## 12.2 在协程中使用阻塞IO
```python
import asyncio
import time

async def blocking_mistake():
    time.sleep(2)   # 阻塞整个事件循环！
    print("完成")

async def main():
    # 错误示例：会阻塞2秒，期间无法处理其他任务
    await blocking_mistake()

# 正确做法：使用run_in_executor
```
## 12.3 异常处理
```python
import asyncio

async def might_fail(should_fail):
    if should_fail:
        raise ValueError("出错了")
    await asyncio.sleep(1)
    return "成功"

async def main():
    tasks = [might_fail(False), might_fail(True)]
    # 使用return_exceptions=True让gather返回异常对象而非抛出
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print(f"捕获到异常: {r}")
        else:
            print(f"结果: {r}")

asyncio.run(main())

```
## 12.4 调试模式
设置环境变量`PYTHONASYNCIODEBUG=1`或在代码中启用：
```python
asyncio.run(main(), debug=True)
```

# 13. 与Web框架结合
```python
# server.py
from fastapi import FastAPI
import asyncio
import httpx   # 异步HTTP客户端，需安装：pip install httpx

app = FastAPI()

async def call_ai(prompt: str):
    async with httpx.AsyncClient() as client:
        # 这里替换为真实的AI服务地址
        resp = await client.post("https://api.openai.com/v1/completions", 
                                 json={"prompt": prompt, "max_tokens": 50},
                                 headers={"Authorization": "Bearer YOUR_KEY"})
        return resp.json()

@app.post("/process")
async def process(prompt: str):
    # 并发调用多个模型（这里模拟调用同一个模型两次）
    results = await asyncio.gather(
        call_ai(prompt),
        call_ai(prompt + " (enhanced)")
    )
    return {"prompt": prompt, "results": results}

# 运行：uvicorn server:app --reload

```


---
参考：https://www.cnblogs.com/wyh-study/p/19730056