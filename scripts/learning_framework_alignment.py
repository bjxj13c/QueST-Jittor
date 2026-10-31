"""Same fixed linear weights and inputs in separate Torch/Jittor processes."""
import argparse
import sys
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--framework',choices=['torch','jittor'],required=True)
    parser.add_argument('--check-grad',action='store_true')
    args=parser.parse_args()
    x=np.array([[1,2],[3,4]],dtype=np.float32)
    weight=np.array([[1,2],[-1,.5],[0,3]],dtype=np.float32)
    bias=np.array([.1,-.2,.3],dtype=np.float32)
    expected=x@weight.T+bias

    if args.framework=='torch':
        import torch
        from torch import nn
        torch.set_num_threads(2)
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(2,3)
            def forward(self,value):return self.linear(value)
        model=SimpleModel().eval()
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(weight))
            model.linear.bias.copy_(torch.from_numpy(bias))
            output=model(torch.from_numpy(x)).numpy()
        if args.check_grad:
            model.train()
            input_tensor=torch.tensor(x,requires_grad=True)
            loss=(model(input_tensor)**2).mean()
            loss.backward()
            gradients={'weight':model.linear.weight.grad.numpy().copy(),
                'bias':model.linear.bias.grad.numpy().copy(),'input':input_tensor.grad.numpy().copy()}
            loss_value=loss.item()
            np.testing.assert_array_equal(model.linear.weight.detach().numpy(),weight)
            np.testing.assert_array_equal(model.linear.bias.detach().numpy(),bias)
    else:
        import jittor as jt
        from jittor import nn
        assert 'torch' not in sys.modules,'Native Jittor process must not import Torch'
        jt.flags.use_cuda=0
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(2,3)
            def execute(self,value):return self.linear(value)
        model=SimpleModel();model.eval()
        model.linear.weight.assign(jt.array(weight))
        model.linear.bias.assign(jt.array(bias))
        with jt.no_grad():output=model(jt.array(x)).numpy()
        if args.check_grad:
            # Jittor eval() stops parameter gradients;train() restores them.
            model.train()
            input_tensor=jt.array(x);input_tensor.start_grad()
            loss=(model(input_tensor)**2).mean()
            derivatives=jt.grad(loss,[model.linear.weight,model.linear.bias,input_tensor])
            gradients={k:v.numpy().copy() for k,v in zip(('weight','bias','input'),derivatives)}
            loss_value=loss.numpy().item()
            np.testing.assert_array_equal(model.linear.weight.numpy(),weight)
            np.testing.assert_array_equal(model.linear.bias.numpy(),bias)
        assert 'torch' not in sys.modules

    np.testing.assert_allclose(output,expected,atol=1e-6,rtol=1e-6)
    print('FRAMEWORK:',args.framework)
    print('INPUT SHAPE:',x.shape)
    print('OUTPUT SHAPE:',output.shape)
    print('OUTPUT:\n',output)
    print('MAX ABS ERROR FROM NUMPY:',float(np.abs(output-expected).max()))
    print('ALIGNMENT: PASSED (fixed weights,no training,CPU only)')
    if args.check_grad:
        # Analytic derivatives for L=mean(Y**2),Y=X@W.T+b.
        dy=2*expected/expected.size
        reference={'weight':dy.T@x,'bias':dy.sum(axis=0),'input':dy@weight}
        np.testing.assert_allclose(loss_value,np.mean(expected**2),atol=1e-5,rtol=1e-6)
        print('LOSS mean(Y**2):',loss_value)
        for name,value in gradients.items():
            np.testing.assert_allclose(value,reference[name],atol=1e-5,rtol=1e-6)
            print(name.upper(),'GRADIENT:\n',value)
            print(name.upper(),'MAX ABS GRADIENT ERROR:',float(np.abs(value-reference[name]).max()))
        print('GRADIENT ALIGNMENT: PASSED;parameters unchanged;no optimizer step')

if __name__=='__main__':main()
