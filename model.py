import torch
class SoftmaxRegression:
    def __init__(self, input_dim , num_classes):

        self.weights=torch.randn(input_dim, num_classes)*0.01
        self.bias=torch.zeros(num_classes)

    def softmax(self, z):
        z_stable=z-torch.max(z, dim=1, keepdim=True)[0]#保证形状是[batch_size, 1], dim=1:每一行中按列求最大值，就是求行的最大值
        exp_z=torch.exp(z_stable)#稳定值, 防止溢出
        return exp_z/torch.sum(exp_z, dim=1, keepdim=True)

    def forward(self, X):
        z=X@self.weights+self.bias
        y_hat=self.softmax(z)
        return y_hat

    def compute_loss(self, y_hat, y_true):
        batch_size=y_hat.shape[0]

        log_probs=torch.log(y_hat[range(batch_size), y_true])#从每个样本的概率预测行中，挑出真实类别的预测概率值
        loss=-torch.mean(log_probs)
        return loss

    def compute_gradient(self, X, y_hat, y_true):
        batch_size=X.shape[0]
        num_classes=self.weights.shape[1]

        y_one_hot=torch.zeros(batch_size, num_classes)#[batch_size, num_classes]
        y_one_hot.scatter_(1, y_true.unsqueeze(1), 1)#[batch_size, num_classes]，是真实标签的one-hot矩阵

        error=y_hat-y_one_hot

        grad_W=(X.T@error)/batch_size#转置[D, N], error:[N, C], 相乘之后和权重矩阵大小一致, D:特征向量的维度

        grad_b=torch.mean(error, dim=0)

        return grad_W, grad_b

    def update_parameters(self, grad_W, grad_b, learning_rate):
        self.weights-=learning_rate*grad_W
        self.bias-=learning_rate*grad_b