import numpy as np

# m序列生成器（移位寄存器）
def m_generator(flag):
    n=len(flag) #移位寄存器长度
    N=2**n-1 #伪随机码周期
    a=np.zeros((1,n-1))
    register=np.append(a,1)
    m=np.zeros(N) #初始化m序列
    newregister=np.append(a,0) #用于记录移位寄存器状态
    for i in range(0,2):
        print(register,newregister)
        newregister[0]=np.mod(np.sum(np.multiply(flag,register)),2)
        for j in range(1,n):
            # print(newregister[j],register[j-1])
            print(register[0])
            newregister[j]=register[j-1]
        register=newregister
        m[i]=register[n-1]
    return m

if __name__ =="__main__":
    flag=np.array([0,0,1,0,0,0,1])# 生成长度2^7-1=127的前导码，可修改
    preamble=m_generator(flag)

