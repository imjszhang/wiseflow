from env import ProjectEnv

# 创建环境
env = ProjectEnv(timeout=2)

# 重置环境
obs, info = env.reset()
print("Initial State:", obs)

# 执行代码
code = """
print("Hello, World!")
x = 5 + 3
print("Result:", x)
"""
state, reward, done, truncated, info = env.step(code)
print("State:", state)
print("Reward:", reward)

# 执行错误代码
code = """
print("This will cause an error")
x = 1 / 0  # Division by zero
"""
state, reward, done, truncated, info = env.step(code)
print("State:", state)
print("Reward:", reward)

# 渲染日志
env.render()

# 关闭环境
env.close()