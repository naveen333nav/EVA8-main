# EVA8

## Part 1

<img src="Assignment 3/Plots/network.png" alt="Alt text" title="Optional title">

</br></br>

> ## Forward Propagation

$$ h1 = i1 _ w1 + i2 _ w2 $$

```math
\begin{aligned}
  a\_h1 = \sigma (h1) \\
 \sigma (h1) = 1/(1+\exp (-h1))
 \end{aligned}
```

```math
o1 = w5 * a\_h1 + w6 * a\_h2
```

```math
a\_o1 = \sigma (o1)
```

```math
E1 = 0.5 * (t1 - a\_o1)^2
```

Similarly calculate for E2

$$ E = E1 + E2 $$

## Backpropagation

> Derivatives :

```math
\frac{\partial E\_total}{\partial E1} = 1
```

```math
      \frac{\partial E\_total}{\partial E2} = 1
```

```math
\frac{\partial a\_o1}{\partial o1} = a\_o1 * (1-a\_o1)
```

```math
 \frac{\partial o1}{\partial a\_h1} = w5
```

```math
\begin{aligned}
\frac{\partial E\_total}{\partial w5} = \frac{\partial E\_total}{\partial a\_o1}* \frac{\partial a\_o1}{\partial o1} * a\_h1  \\
\frac{\partial E\_total}{\partial w8} = \frac{\partial E\_total}{\partial a\_o2}* \frac{\partial a\_o2}{\partial o2}* a\_h2 \\
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial h1}{\partial w1} = i1 \\
\frac{\partial h2}{\partial w4} = i2 \\
\end{aligned}


```

```math
\frac{\partial a\_h1}{\partial h1} = a\_h1 * (1-a\_h1)
```

```math
\frac{\partial E\_total}{\partial o1} = \frac{\partial E\_total}{\partial E1} *
        \frac{\partial E1}{\partial a\_o1}  * \frac{\partial a\_o1}{\partial o1}
```

```math
\frac{\partial E\_total}{a\_h1} = \frac{\partial E\_total}{o1} * \frac{\partial o1}{\partial a\_h1} + \frac{\partial E\_total}{o2} * \frac{\partial o2}{\partial a\_h1}
```

```math
\begin{aligned}
\frac{\partial E\_total}{\partial w1} = \frac{\partial E\_total}{\partial a\_h1} * \frac{\partial a\_h1}{\partial h1} * \frac{\partial h1}{\partial w1} \\
\frac{\partial E\_total}{\partial w4} = \frac{\partial E\_total}{\partial a\_h2} * \frac{\partial a\_h2}{\partial h2} * \frac{\partial h2}{\partial w4}
\end{aligned}
```

> Plots for different values of learning rate

<div>
<img
  src="Assignment 3/Plots/learning rate 0_1.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

<img
  src="Assignment 3/Plots/learning rate  0_2.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

<img
  src="Assignment 3/Plots/learning rate 0_5.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

<img
  src="Assignment 3/Plots/learning rate 0_8.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

<img
src="Assignment 3/Plots/leanring rate 2.png"
alt="Alt text"
title="Optional title"
style="display: inline-block; margin: 0 auto; max-width: 300px">

</div>

<br> </br>

> ## Part 2

```python
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]           3,904
       BatchNorm2d-2           [-1, 32, 28, 28]              64
         MaxPool2d-3           [-1, 32, 14, 14]               0
           Dropout-4           [-1, 32, 14, 14]               0
            Conv2d-5           [-1, 64, 14, 14]           2,112
       BatchNorm2d-6           [-1, 64, 14, 14]             128
         MaxPool2d-7             [-1, 64, 7, 7]               0
           Dropout-8             [-1, 64, 7, 7]               0
            Conv2d-9            [-1, 128, 7, 7]           8,320
        AvgPool2d-10            [-1, 128, 1, 1]               0
           Linear-11                   [-1, 10]           1,290
================================================================
Total params: 15,818
Trainable params: 15,818
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.77
Params size (MB): 0.06
Estimated Total Size (MB): 0.83
----------------------------------------------------------------
```

### The above represents network summary
