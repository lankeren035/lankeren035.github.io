- 输入图像：
	$$
	x= [x_1 , x_2]
	$$
	

- 线性层：
	$$
	y=w @ x+b \space \space \space , \space \space \space w=[w_1 , w_ 2]
	$$
	

- 预测图：
	$$
	y=[x_ 1 \times w_ 1+b \space \space, \space \space  x_ 2 \times w_ 2+b]
	$$
	

- loss:
	$$
	l=[(x_ 1 \times w_ 1+b)-x_ 1]^2 + [(x_ 2 \times w_ 2+b)-x_ 2]^2
	$$
	



- 计算图：从x到l的一张有向无环图，记录了每一步算子用于反向求导（求导公式，开销小）以及中间变量（开销大）：
	$$
	\begin{aligned}
	\textcolor{red}{\text{Lin}_1} &: y_1=x_ 1 \textcolor{red}{\times} w_ 1 \textcolor{red}{+} b \\\\
	\textcolor{red}{\text{Lin}_2} &: y_2 = x_ 2 \textcolor{red}{\times} w_ 2 \textcolor{red}{+} b \\\\
	\textcolor{red}{\text{Sub}_ 1} &: \textcolor{red}{t_1} = y_1 \textcolor{red}{-} x_1 \\\\
	\textcolor{red}{\text{Sub}_ 2} &: \textcolor{red}{t_2} = y_2 \textcolor{red}{-} x_2 \\\\
	\textcolor{red}{\text{Pow}_ 1} &: z_1 = t_ 1^ \textcolor{red}{2} \\\\
	\textcolor{red}{\text{Pow}_ 2} &: z_2 = t_ 2^ \textcolor{red}{2} \\\\
	\textcolor{red}{\text{Sum}} &: l= z_ 1 \textcolor{red}{+} z_ 2
	\end{aligned}
	$$

- 反向传播时：需要更新：$x_ 1, x_ 2$ ， 所以要求loss对于他们的梯度：$\frac{\partial l}{\partial x_ 1} \space, \space \frac{\partial l}{\partial x_ 2}$ :
	$$
	\begin{aligned}
		\frac{\partial l}{\partial x_ 1} &= \frac{\partial l}{\partial z_ 1} \frac{\partial z_ 1}{\partial t_ 1} \frac{\partial t_ 1}{\partial y_ 1} \frac{\partial y_ 1}{\partial x_ 1} + \frac{\partial l}{\partial z_ 1} \frac{\partial z_ 1}{\partial t_ 1} \frac{\partial t_ 1}{\partial x_ 1} \\\\
		&=1 \cdot 2\textcolor{red}{t_ 1} \cdot 1 \cdot \textcolor{red}{w_ 1} + 1 \cdot 2\textcolor{red}{t_ 1} \cdot (-1)  \\\\
		\frac{\partial l}{\partial x_ 2} &= \frac{\partial l}{\partial z_ 2} \frac{\partial z_ 2}{\partial t_ 2} \frac{\partial t_ 2}{\partial y_ 2} \frac{\partial y_ 2}{\partial x_ 2} + \frac{\partial l}{\partial z_ 2} \frac{\partial z_ 2}{\partial t_ 2} \frac{\partial t_ 2}{\partial x_ 2} \\\\
		&=1 \cdot 2\textcolor{red}{t_ 2} \cdot 1 \cdot \textcolor{red}{w_ 2} + 1 \cdot 2\textcolor{red}{t_ 2} \cdot (-1)
	\end{aligned}
	$$
	





#### 裁剪之后：

