import torch

class MissingModalityHandler:
    """
    Representa explícitamente las modalidades faltantes mediante
    un tensor imputado, una máscara binaria y un indicador de calidad.
    """
    @staticmethod
    def handle_missing(tensors_dict: dict, expected_modalities: list, default_dims: dict):
        """
        Garantiza que todas las modalidades esperadas estén presentes en el diccionario.
        Si falta una, se rellena con ceros y la máscara se pone a 0.
        
        Returns:
            dict con "tensors" y "masks"
        """
        processed = {"tensors": {}, "masks": {}}
        
        # Asumimos que al menos una modalidad presente define el batch_size y device
        batch_size = None
        device = None
        for k, v in tensors_dict.items():
            if v is not None:
                batch_size = v.shape[0]
                device = v.device
                break
                
        if batch_size is None:
            raise ValueError("Al menos un tensor válido debe estar presente.")
            
        for mod in expected_modalities:
            if mod in tensors_dict and tensors_dict[mod] is not None:
                processed["tensors"][mod] = tensors_dict[mod]
                processed["masks"][mod] = torch.ones(batch_size, 1, device=device)
            else:
                dim = default_dims.get(mod, 1)
                processed["tensors"][mod] = torch.zeros(batch_size, dim, device=device)
                processed["masks"][mod] = torch.zeros(batch_size, 1, device=device)
                
        return processed
