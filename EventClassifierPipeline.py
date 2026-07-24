import torch
import pca
# =================================================
# Important: classification layers, event breakdown
# =================================================

class EventClassifierPipeline:

    def __init__(self, model_traced_path, onlyACDVeto=True, random_forest_path=None, lookup_path=None, three_class=False):
        self.three_class = three_class

        if three_class:
            print("Using 3-class PointNet model (PointNetModels/pointnet3C.py)")
            from PointNetModels.pointnet3C import PointNet
        else:
            print("Using 2-class PointNet model (PointNetModels/pointnet2C.py)")
            from PointNetModels.pointnet2C import PointNet

        if not onlyACDVeto:
            if random_forest_path is not None:
                print(f"Loading RF model from {random_forest_path}...")
                self.pca_classifier = pca.VegaClassifier(random_forest_path)
            elif lookup_path is not None:
                print(f"Loading lookup table from {lookup_path}...")
                self.pca_classifier = pca.SimpleClassifier(lookup_path)
            else:
                print("No BKG classifier provided — events will return 0.0")
                self.pca_classifier = None
        else:
            self.pca_classifier = None

        print(f"Loading TorchScript model from {model_traced_path}...")
        self.model = PointNet(add_nhits=False)
        state_dict = torch.load(model_traced_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()  

    def extract_hit_data(self, event, detId=None):
        ''' 
        Extracts hit data from the event and returns a tensor of shape [1, 4, nhits].
        The first dimension is the batch size (1), the second dimension is x,y,z,E (4),
        and the third dimension is the number of hits (nhits). Returns None if no hits found.
        If detId is specified, only hits from that detector type are extracted.
        '''
        nhits = event.GetNHTs()
        if nhits == 0:
            return None

        data = torch.zeros([1, 4, nhits])
        if detId is None:
            for i in range(nhits):
                data[0, 0, i] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, i] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, i] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, i] = event.GetHTAt(i).GetEnergy()
            return data

        n_selected = 0
        for i in range(nhits):
            if event.GetHTAt(i).GetDetectorType() == detId:
                data[0, 0, n_selected] = event.GetHTAt(i).GetPosition().X()
                data[0, 1, n_selected] = event.GetHTAt(i).GetPosition().Y()
                data[0, 2, n_selected] = event.GetHTAt(i).GetPosition().Z()
                data[0, 3, n_selected] = event.GetHTAt(i).GetEnergy()
                n_selected += 1

        if n_selected > 0:
            return data[:, :, :n_selected]
        return None
    
    # "L1"
    def signal_background_classifier(self, event, debug=False, onlyACDVeto=True, thr=0.99, log_file=None):
        """
        First layer: Checks if the event is background or a signal.
        Returns the classification (str) and the probability (float) of that classification.
        """
        
        # If for some reason the event is None, return unknown with probability 1.0
        if not event:
            if debug and log_file:
                log_file.write(f"WARNING (L1): Received None event\n")
                print("WARNING check log.txt")
            return "UN", 1.00
        
        # 0 hits is unknown
        nhits = event.GetNHTs()
        if nhits == 0:
            return "UN", 1.00

        # Check for ACD hits. If there are any, classify as MU with high probability.
        nhits_ACD  = 0
        for i in range(nhits):
            if event.GetHTAt(i).GetDetectorType() == 4:
                nhits_ACD += 1
        if nhits_ACD > 0:  
            return "MU", 0.99  

        # If there are no ACD hits, use PCA to classify as SIGNAL or MU.
        if not onlyACDVeto:
            data = self.extract_hit_data(event, 1)
            prob = pca.analyze(data, event.GetTotalEnergyDeposit(), rf=self.pca_classifier, thr=thr)
            if prob > 0.5:
                return "SIGNAL", prob
            return "MU", 1.-prob
        else:
            return "SIGNAL", 1.00

    #"L2"
    def type_of_signal(self, event, debug, log_file=None):
        """Second layer: Checks if the event is a Photoelectric effect.

        If not, use PointNet to discriminate the event topology:
        - binary model  -> Compton vs Pair
        - 3-class model -> Compton vs Pair vs PH

        Returns the classification (str) and the probability (float) of that type.
        """

        # Should never enter here, given that the first layer already checks for None events, but just in case.
        if not event:
            if debug and log_file:
                log_file.write(f"WARNING (L2): Received None event\n")
                print("WARNING check log.txt")
            return "UN", 1.00
        
        nhits = event.GetNHTs()
        
        # Should never enter here, given that the first layer already checks for 0 hits, but just in case.
        if nhits == 0:
            return "UN", 1.00

        # 1.Cut for  Photoelectric effect (PHOT), less than 2 hits is likely a photoelectric effect. Return 'PH' with probability 0.50 (uncertain).
        if not nhits > 2:
            return "PH", 0.50  # 'PH' for Photoelectric

        # 2. If not Photoelectric, extract hit data and execute PointNet
        data_input = self.extract_hit_data(event)
        
        # 3. Dispatch to the head matching the loaded model
        if self.three_class:
            return self._type_of_signal_3class(data_input)
        else:
            return self._type_of_signal_binary(data_input)

    def _type_of_signal_binary(self, data_input):
        """PointNet binary head: single scalar logit -> sigmoid + threshold at 0."""
        # Run the PointNet model to classify between Compton and Pair Production
        with torch.no_grad():
            logits, _ = self.model(data_input)
            prob = torch.sigmoid(logits).item()

        # Return the classification and probability based on the logits
        if logits >= 0:
            return "PA", prob  # 'PA' for Pair Production
        elif logits < 0:
            return "CO", 1.0 - prob  # 'CO' for Compton Scattering

        # Fallback case, should not be reached
        return "UN", 1.00

    def _type_of_signal_3class(self, data_input):
        """PointNet 3-class head: 3 logits (Compton, Pair, Photoelectric) -> softmax + argmax."""
        label_map = {0: "CO", 1: "PA", 2: "PH"}  # same order used in training
        
        # Run the PointNet model to classify between Compton, Pair and Photoelectric
        with torch.no_grad():
            logits, _ = self.model(data_input)            # shape [1, 3]
            probs = torch.softmax(logits, dim=1)           # softmax
            pred_idx = torch.argmax(probs, dim=1).item()   # class with highest probability
            prob = probs[0, pred_idx].item()

        return label_map[pred_idx], prob

