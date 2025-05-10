import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module): 
    def __init__(self, size_average=False, loss_lambda=0.5):
        '''
        Lk = Tk max(0, m+ - ||vk||)2 + λ (1 - Tk) max(0, ||vk|| - m-)2      (4)
        '''
        super(MarginLoss, self).__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda
        
    def forward(self, inputs, labels): 
        L_k = labels * F.relu(self.m_plus - inputs)**2  + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
        return L_k.mean() if self.size_average else L_k.sum()
    
    
class CapsuleLoss(nn.Module):
	def __init__(self, loss_lambda=0.5, recontruction_loss_scale=5e-4, size_average=False):
		'''
		Combined loss: L_margin + L_reconstruction (SSE was used as reconstruction)
		
		Params:
		- recontruction_loss_scale: 	param for scaling down the the reconstruction loss.
	    - size_average:		    if True, reconstruction loss becomes MSE instead of SSE.
		'''
		super(CapsuleLoss, self).__init__()
		self.size_average = size_average
		self.margin_loss = MarginLoss(size_average=size_average, loss_lambda=loss_lambda)
		self.reconstruction_loss = nn.MSELoss(size_average=size_average)
		self.recontruction_loss_scale = recontruction_loss_scale

	def forward(self, inputs, labels, images, reconstructions):
		margin_loss = self.margin_loss(inputs, labels)
		reconstruction_loss = self.reconstruction_loss(reconstructions, images)
		caps_loss = (margin_loss + self.recontruction_loss_scale * reconstruction_loss)

		return caps_loss   