from mjlab.rl import MjlabOnPolicyRunner

class Obs124Runner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # A standard single MLP with 124 inputs; no phase branch or extra state.
        for parameter in self.alg.actor.distribution.parameters():
            parameter.requires_grad_(False)
    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        return super().learn(num_learning_iterations,init_at_random_ep_len=False)
