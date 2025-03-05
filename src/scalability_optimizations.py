import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ScalabilityOptimizer:
    def __init__(self, model):
        self.model = model
        
    def structured_pruning(self, amount=0.2):
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                if isinstance(module, nn.Conv2d):
                    prune_amount = amount
                elif isinstance(module, nn.Linear):
                    prune_amount = amount
                else:
                    prune_amount = amount
                prune.ln_structured(module, name='weight', amount=prune_amount, n=1, dim=0)
                prune.remove(module, 'weight')
                
    def quantize_model(self, calibration_data, device="cuda"):
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        torch.quantization.prepare(self.model, inplace=True)
        with torch.no_grad():
            for batch in calibration_data:
                self.model(batch.to(device))
        torch.quantization.convert(self.model, inplace=True)
        
    def apply_knowledge_distillation(self, student_model, teacher_model, train_loader,
                                   temperature=3.0, alpha=0.5, device="cuda"):
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(student_model.parameters())
        
        student_model.to(device)
        teacher_model.to(device)

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_logits = teacher_model(batch) / temperature
                teacher_probs = nn.functional.softmax(teacher_logits, dim=1)
            
            # Train student
            student_logits = student_model(batch) / temperature
            student_probs = nn.functional.log_softmax(student_logits, dim=1)
            
            loss = criterion(student_probs, teacher_probs) * alpha + nn.functional.cross_entropy(student_logits, torch.argmax(teacher_probs, dim=1)) * (1 - alpha)
            loss.backward()
            optimizer.step()
