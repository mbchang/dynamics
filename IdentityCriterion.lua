require 'nn'

-- Used when we don't want to pass the gradients through

local IdentityCriterion, parent = torch.class('nn.IdentityCriterion', 'nn.Criterion')

function IdentityCriterion:__init(motionScale)
   parent.__init(self)
end

function IdentityCriterion:updateOutput(input, target)
   -- loss = 0
   self.output = 0
   return self.output
end

function IdentityCriterion:updateGradInput(input, target)
   -- grad is 0
   self.gradInput:resizeAs(input):fill(0)  -- self.gradInput was initialized in the parent class
   return self.gradInput
end
