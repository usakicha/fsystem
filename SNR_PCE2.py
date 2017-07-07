import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

NI=21
#定义函数

def fftshift(x):
    y1=tf.slice(x,[0,0,0],[-1,540,540])
    y2=tf.slice(x,[0,0,540],[-1,540,540])
    y3=tf.slice(x,[0,540,0],[-1,540,540])
    y4=tf.slice(x,[0,540,540],[-1,540,540])
    o1=tf.stack([y4,y3],axis=2)
    o2=tf.stack([y2,y1],axis=2)
    o1=tf.reshape(o1,shape=(-1,540,1080))
    o2=tf.reshape(o2,shape=(-1,540,1080))
    O=tf.stack([o1,o2],axis=1)
    O=tf.reshape(O,shape=(-1,1080,1080))
    return O



 
def LENS1(f1,x,y,sita,lamda):
    lens10=np.exp(-1j*2*np.pi/(lamda*2*f1)*(np.power(x,2)*np.power(np.cos(sita),2)+np.power(y,2)))
    len1=np.angle(lens10)
    len1=np.mod(len1,2*np.pi)
    return len1



def LENS2(f2,x,y,sita,lamda):
    lens20=np.exp(-1j*2*np.pi/(lamda*2*f2)*(np.power(x,2)*np.power(np.cos(sita),2)+np.power(y,2)))
    len2=np.angle(lens20)
    len2=np.mod(len2,2*np.pi)
    return len2


#初始化
lamda=0.633  
sita=6*np.pi/180
p=8
f1=340e3
f2=170e3
M=1080
N=1080
l2=f2*f1/(f1-f2)
x=np.linspace(-p*M/2,p*M/2,M)
y=np.linspace(-p*N/2,p*N/2,N)
[x,y]=np.meshgrid(x,y)
du=1/M/p
dv=1/N/p
u=np.multiply((range(int(-M/2),int(M/2))),du)
v=np.multiply((range(int(-N/2),int(N/2))),dv)
[u,v]=np.meshgrid(u,v)
u0=np.sin(sita)/lamda
x1=f1*np.sin(sita)
x2=l2*np.sin(sita)
H1=np.multiply(np.exp(1j*2*np.pi*f1/lamda*np.sqrt(1-np.power((u+u0),2)*np.power(lamda,2)-np.power(v,2)*np.power(lamda,2))),np.exp(1j*2*np.pi*u*x1))
H2=np.multiply(np.exp(1j*2*np.pi*l2/lamda*np.sqrt(1-np.power((u+u0),2)*np.power(lamda,2)-np.power(v,2)*np.power(lamda,2))),np.exp(1j*2*np.pi*u*x2))
len1=LENS1(f1,x,y,sita,lamda)
len2=LENS2(f2,x,y,sita,lamda)
coe=[1,1,1,1,1]
Method=2


#读取图片
img=[]
for i in range (1,22):
    img.append(cv2.imread(str(i)+'.bmp',0))

img=np.array(img,dtype=np.double)

#求滤波器初始值
lens1=np.exp(-1j*2*np.pi/(lamda*2*f1)*(np.power(x,2)*np.power(np.cos(sita),2)+np.power(y,2)))
IMG=np.exp(-1j*(img[0]/255*2*np.pi))
IMG=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(IMG)))
IMG=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.multiply(IMG,H1))))
IMG=np.multiply(IMG,lens1)
IMG=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(IMG)))
IMG=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.multiply(IMG,H1))))
angle_h=np.angle(IMG)
step_h=255
binary_1=np.mod(angle_h,2*np.pi)
binary_2=np.round(binary_1/(2*np.pi/step_h))*(2*np.pi/step_h)
h=binary_2



#求所有图的频谱

A=img/255*2*np.pi
SLM1=np.mod(A-len1,2*np.pi)


out_SLM1=np.exp(-1j*SLM1)
U12=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(out_SLM1)))
U21=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.multiply(U12,H1))))
abs_U21=np.abs(U21)

Input=tf.placeholder(shape=(None,1080,1080),dtype=tf.complex64) #Input=U21
core=tf.constant(1,shape=(7,7,1,1),dtype=tf.float32)
hh=tf.Variable(h,dtype=tf.float32)
tfSLM=(hh-len2)-tf.floor((hh-len2)/2*np.pi)*2*np.pi

hhh=tf.complex(tfSLM,0.)


U31=tf.multiply(Input,tf.exp(-1j*hhh))
U32=fftshift(tf.fft2d(fftshift(U31)))
Uout1=fftshift(tf.ifft2d(fftshift(tf.multiply(U32,H2))))
Uout2=tf.pow((tf.abs(Uout1)),2)


#评价模块

U2=tf.reshape(Uout2,shape=(-1,1080,1080,1))
R=tf.nn.conv2d(U2,core,[1, 1, 1, 1], padding='SAME')
rr=tf.reshape(R,shape=(-1,1080*1080))
#Rmax=tf.reduce_max(rr,1)
#RR=Rmax/10


U=tf.reshape(Uout2,shape=(-1,1080*1080))
a=tf.reduce_max(U,1)
Rmax=[0]*NI
place=tf.argmax(U,axis=1)
place2=tf.to_int32(place)
Ec_half=[0]*NI
half_sum2=[0]*NI
SNR=[0]*NI
for i in range(NI):
    Rmax[i]=rr[i][place2[i]]
    b=tf.where(U[i]<(0.5*a[i]))
    half_sum=tf.size(b)
    half_sum2[i]=tf.to_float(half_sum)
    Ec_half[i]=tf.subtract(tf.reduce_sum(tf.pow(tf.add((tf.negative(tf.nn.relu(tf.negative(tf.subtract(U[i],0.5*a[i]))))),0.5*a[i]),2)),(tf.multiply(tf.pow((0.5*a[i]),2),(1166400-half_sum2[i]))))
    SNR[i]=a[i]/tf.sqrt(Ec_half[i]/half_sum2[i])


RR=Rmax
SNR2=tf.subtract(SNR,tf.nn.relu(tf.subtract(SNR,200)))
loss=tf.concat([RR,SNR2],axis=0)
#loss=RR
train=tf.train.GradientDescentOptimizer(learning_rate=500).minimize(tf.negative(tf.log(loss)))







sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()


step=1
for i in range(3500):
    sess.run(train,feed_dict={Input:U21})
    step=step+1
    if i%10 == 0:
        print('step=',step)
        print('R=',sess.run(Rmax,feed_dict={Input:U21}))
        print('SNR=',sess.run(SNR,feed_dict={Input:U21}))


sio.savemat('2f.mat',{'FIL':sess.run(hh)})







import matplotlib.pyplot as plt
b=sess.run(Uout2,feed_dict={Input:U21})
plt.subplot(111),plt.imshow(b[18],'gray'),plt.title('original')
plt.show()
