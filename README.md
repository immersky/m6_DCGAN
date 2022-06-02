个人学习DCGAN

阅读了论文和相关资料，借助pytorch的教程理解

参考https://zhuanlan.zhihu.com/p/57348649

代码实现参考pytorch官方DCGAN教程https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan

pytorch官网推荐的GAN项目https://github.com/nashory/gans-awesome-applications

使用的数据集下载:

链接：https://pan.baidu.com/s/1nC_i6ZCiDbCUcWDOCcWrdQ?pwd=tmdk 



DCGAN在原始GAN的生成器和鉴别器加入了CNN的结构，但舍弃了CNN的pooling层（池化层），生成器则把卷积层换成了微步幅卷积层，使用BN，激活函数中间用ReLu,最后用Tanh,使用Adam优化器，在较深的结构删除全连接层

如果理解了GAN和CNN，DCGAN非常好理解，只有一点需要扩充:微步幅卷积

# 微步卷积(Fractionally strided convolution)

微步幅卷积（Fractionally strided convolution）又称转置卷积,以及Transposed Convolution,以及被错误的叫做反卷积(Deconvolution)。

参考https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

微步卷积的数学推导https://arxiv.org/abs/1603.07285

首先知悉一下上采样，下采样的概念

下采样实际上就是缩小图像，主要目的是为了使得图像符合显示区域的大小，生成对应图像的缩略图。比如说在CNN中得池化层或卷积层就是下采样。不过卷积过程导致的图像变小是为了提取特征，而池化下采样是为了降低特征的维度。

 上采样（UpSampled），际上就是放大图像，指的是任何可以让图像变成更高尺寸，上采样有3种常见的方法：双线性插值(bilinear)，微步幅卷积（Fractionally strided convolution），反池化(Unpooling)，



反卷积的叫法并不好，DCGAN论文中称其为一个错误，信号处理中的去卷积是卷积运算的逆运算，而这里则不是正向卷积的完全逆过程，用一句话来解释：

微步幅卷积是一种特殊的正向卷积，先按照一定的比例通过补0 来扩大输入图像的尺寸，接着旋转卷积核，再进行正向卷积。

过程如下

在正常卷积中，我们这样定义：用C代表卷积核，input为输入图像，out为输出图像。经过卷积（矩阵乘法）后，我们将input从大图像下采样为小图像output。这种矩阵乘法实现遵循$C*input=output$

如下图的例子

![conv1](https://user-images.githubusercontent.com/74494790/171611423-ad369814-7547-4ec8-a6fc-f2bc3e4e028a.png)


此时，若用卷积核对应稀疏矩阵的转置$C^T(16*4)$乘以输出的平展4*1矩阵，得到的结果和输入时的尺寸相同，如下图


![conv2](https://user-images.githubusercontent.com/74494790/171611465-96c75697-f7d3-4bba-af78-611b22c3a41c.jpg)



缺陷：棋盘效应,参考:https://distill.pub/2016/deconv-checkerboard/

​	棋盘效应是由于转置卷积的“不均匀重叠”（Uneven overlap）的结果。使图像中某个部位的颜色比其他部位更深。尤其是当卷积核（Kernel）的大小不能被步长（Stride）整除时，微步幅卷积就会不均匀重叠。虽然原则上网络可以通过训练调整权重来避免这种情况，但在实践中神经网络很难完全避免这种不均匀重叠，这边不多说了

放两个直观的图，源自上面的网址

![notgood1](https://user-images.githubusercontent.com/74494790/171611483-8d122440-853b-410b-be8c-5dbbe313095d.png)
![notgood2](https://user-images.githubusercontent.com/74494790/171611485-ae8789b0-f258-441f-98bd-fc2a4a841396.png)




微步幅卷积尺寸推导

输入尺寸i

卷积核大小k

步幅s

边界扩充p

输出尺寸o

则$o=s*(i-1)-2p+k$,如果和原尺寸不一样再加常数，如果o,s字母互换可以发现它成了卷积的尺寸计算公式

# DCGAN网路结构

论文中给出了一张生成器的结构图

![ggraph](https://user-images.githubusercontent.com/74494790/171611501-04e0c826-6110-45b5-919b-fee92a8a3744.png)


# pytorch官方教程描述的一些代码实现细节

DCGAN 论文中，作者指定所有模型权重都应从 mean=0 、 stdev=0.02的正态分布随机初始化。 weights_init 函数将初始化的模型作为输入，并重新初始化所有卷积、卷积转置和批量归一化层以满足此标准。此函数在初始化后立即应用于模型。

```python
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```







鉴别器由跨步卷积层、BN层和 LeakyReLU 激活组成，输入是 3x64x64 的图像，输出是输入来自真实数据分布的标量概率。生成器由转置卷积层、BN层和 ReLU 激活组成，输入是从标准正态分布中提取的潜在向量 z，输出是 3x64x64的RGB 图像。 转置卷积层允许将潜在向量转换为与图像形状相同的体积。在DCGAN论文中，作者还给出了一些关于如何设置优化器、如何计算损失函数以及如何初始化模型权重的提示

生成器 G 旨在将潜在空间向量 (z) 映射到数据空间。由于我们的数据是图像，因此将 z 转换为数据空间意味着最终创建与训练图像大小相同的 RGB 图像（即 3x64x64）。在实践中，这是通过一系列跨步二维转置卷积层实现的，每个层都与一个 2维 BN层和一个 ReLU 激活组合。生成器的输出通过 tanh 函数馈送以将其返回到 [-1,1] 的输入数据范围。值得注意的是，在 conv-transpose 层之后存在BN函数，因为这是 DCGAN 论文的重要贡献。这些层有助于训练期间的梯度流动。 

```python
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution                              #输入大小为100的高斯噪声
            nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 8,kernel_size= 4,stride=1,padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),                                            #论文提到，卷积后使用BN
            nn.ReLU(True),                                                      #论文提到，生成器使用ReLU
            # state size. (ngf*8) x 4 x 4  输出尺寸4=1*(1-1)+4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4,2, 1, bias=False),           #通过ngf*4个4x4xngf*8的微步卷积核
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8  输出尺寸8=2*(4-1)+4-2
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16 输出尺寸为16=2*（8-1）-2+4
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32           输出尺寸为32=2*(16-1)-2+4
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64            输出尺寸为64=2*(16-1)-2+4,三通道
        )

    def forward(self, input):
        return self.main(input)
```





如前所述，鉴别器 D 是一个二元分类网络，它将图像作为输入并输出输入图像是真实的（而不是假的）的标量概率。在这里，D 采用 3x64x64 输入图像，通过一系列 Conv2d、BatchNorm2d 和 LeakyReLU 层对其进行处理，并通过 Sigmoid 激活函数输出最终概率。如果需要，这种架构可以根据实际问题扩展更多层，但使用跨步卷积、BN层 和 LeakyReLUs 具有重要意义。 DCGAN 论文提到使用跨步卷积而不是池化来下采样是一个很好的做法，因为它可以让网络学习自己的池化函数。BN层和LeakyReLU函数也促进了健康的梯度流，这对于 G 和 D 的学习过程至关重要。

```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),                #判别器，是个挺基础的CNN，这里不推导尺寸了
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),             #论文提到，不要全连接
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

```



损失函数以及优化器

通过对 D 和 G 的设置，我们可以指定它们如何通过损失函数和优化器进行学习。我们将使用 PyTorch 中定义的二元交叉熵损失 ([BCELoss](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23torch.nn.BCELoss)) 函数：

$l(x,y)=L=\{l_1,.......,l_n\},l_n=-[y_n *logx_n+(1-y_n)*log(1-x_n)]$

请注意此函数如何提供目标函数中的两个对数分量的计算（即 ![[公式]](https://www.zhihu.com/equation?tex=log%28D%28x%29) ) 和 ![[公式]](https://www.zhihu.com/equation?tex=log%281-D%28G%28z%29%29%29) )。我们可以指定 BCE 方程的哪一部分与 y 输入一起使用。这是在即将推出的训练循环中完成的，但重要的是要了解我们如何仅通过更改 y（即 GT 标签）来选择我们希望计算的组件。

接下来，我们将真实标签定义为 1，将假标签定义为 0。这些标签将在计算 D 和 G 的损失时使用，这也是原始 GAN 论文中使用的约定。最后，我们设置了两个单独的优化器，一个用于 D，一个用于 G。如 DCGAN 论文中所述，两者都是 Adam 优化器，学习率为 0.0002，Beta1 = 0.5。为了跟踪生成器的学习进程，我们将生成一组**固定的潜在向量**（即 fixed_noise)（见代码[Import](https://zhuanlan.zhihu.com/p/379330500/edit#Import)部分，固定了随机种子，因此每次生成的随机向量都是固定的），这些向量是从高斯分布中提取的。在训练循环中，我们会周期性地将这个 fixed_noise 输入到 G 中，在迭代过程中，我们将看到从噪声中形成的图像。

```python
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device) # 64个100*1*1的z向量（小猴子注）

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

训练:

代码中的参数说明

Loss_D:(log(D(x)) + log(1 - D(G(z)))*l**o**g*(*D*(*x*))+*l**o**g*(1−*D*(*G*(*z*))))的损失

Loss_G log(D(G(z)))*l**o**g*(*D*(*G*(*z*)))的损失

**D(x)** - 所有真实批次的鉴别器的平均输出（跨批次）。这应该从接近 1 开始，然后在 G 变得更好时理论上收敛到 0.5。

**D(G(z))** - 所有假批次的鉴别器的平均输出。第一个数字在 D 更新之前，第二个数字在 D 更新之后。这些数字应该从接近 0 开始，随着 G 变得更好而收敛到 0.5。

```python
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

