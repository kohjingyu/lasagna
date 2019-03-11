import copy
import numpy as np
class result_storage():
    def __init__(self,is_test,batch_size,num_classes,num_batches):
        self.is_test = is_test
        self._TPlist={}
        self._FPlist={}  #format =>   TPlist [epochnumber][pvalue]["TP or TN or FN or FP"]
        self._TNlist={}
        self._FNlist={}
        self._train_losses={} # sums up all train losses. Key = epochnumber
        self._val_losses = {} # sums up all validation losses Key = epochnumber
        # of note is that it's called val losses here but it might not actually be so.
        # it could be test loss
        self._val_accuracies = {} # format => val_accuracies[epochnumber][threshold]
        # same goes for this.
        self._num_batches = num_batches
        self._classwise_acc={} #format=> classwise_accuracy[epochnumber][classnumber]
        self._batch_size = batch_size # for internal calculations
        self._num_classes = num_classes # for internal calculations.
    
    
    def accuracy_calculation_epoch(self,epoch):
        # calculate accuracy. called before you obtain accuracies.
        # ONLY CALL THIS ONCE. don't call before every obtain for the same epoch. you're just screwing it up now.
        # Shaun will probably cry. he doesn't want to rewrite this again.
        #did you know he rewrote this 3 times because he thought the structure was too messy
        # anyway its still messy. what a failure.
        for threshold_value in self._val_accuracies[epochs].keys():
                self._val_accuracies[epochs][threshold_value]=self._val_accuracies[epochs][threshold_value]/(self._num_classes*self._batch_size*self._num_batches)

    
    def classwise_acc(self,epoch,classnumber):
        """
        returns a dictionary of the TP,FP,TN,FN
        for this particular class, of this particular epoch.
        Return value: dictionary[threshold] 
        Dictionary is freshly created.
        """
        return_dict = {}
        for threshold in self._classwise_acc[epoch].keys():
            return_dict[threshold] = self._classwise_acc[epoch][threshold][classnumber]
        return return_dict
    
    def store_train_loss(self,epoch,results):
        """Where results refers to some loss function's output."""
        if epoch not in self._train_losses.keys():
            self._train_losses[epoch] = []
        self._train_losses[epoch].append(results.item())
    
    def train_losses(self,epoch):
        """
        returns train loss for a particular epoch
        return value: some number.
        """ 
        
        return self._train_losses[epoch]
    
    def losses(self,epoch):
        """
        returns validation losses for a particular epoch
        return value: some number.
        """
        return self._val_losses[epoch] # generalised name again
    
    def accuracies(self,epoch):
        """
        returns validation or test accuracies for a particular epoch.
        does not have classwise segmentation.
        return value: dictionary[threshold values]
        """    
        return self._val_accuracies[epoch] #generalised name again
    
    
    def TP_FPplz(self,epoch):
        """
        Please note that the values are only with regards to test or validation.
        Although.. you could use this class to store training data too, by pretending it's test or validation
        data. You nasty.
        Returns a dictionary, with the following keys of 
        TP, FP, FN, FP in that dictionary, and then a nested dictionary of pvalues.
        It represents the number of TP, FP , FN and FP of that class of that epoch.
        return value: dict["FP" or "TP" or whatever] [pvalue threshold]
        """
        return copy.deepcopy({"TP": self._TPlist[epoch],"FP": self._FPlist[epoch],"FN":self._FNlist[epoch],"TN": self._TNlist[epoch]})
        
    def data_entry(self,answers,results, epochs,output):
        """ requires the following:
            # The correct answers (answers)
            # the latest validation results (to store the loss)
            # the current epoch number.
            #IF YOU WRITE A DUPLICATE NUMBER IT WILL OVERWRITE A PREVIOUS ENTRY.
            # output of the latest evaluation, no you don't need to detach it, please don't.
        """
        if epochs not in self._TPlist.keys():
            self._TPlist[epochs]={}
            self._FPlist[epochs]={}
            self._TNlist[epochs]={}
            self._FNlist[epochs]={}
            #append epochs
        if epochs not in self._classwise_acc.keys():
            self._classwise_acc[epochs] = {}
            
        if epochs not in self._val_losses.keys():
            self._val_losses[epochs] = results.item() #add an entry for this epoch
        else:
            self._val_losses[epochs] = results.item() + self._val_losses[epochs] # sum up this batch's loss.
        if epochs not in self._val_accuracies.keys():
            self._val_accuracies[epochs] ={} #add an entry for this epoch
            
        for threshold in range(0,105,5): # 0-100 percent certainty values.
            predicted = np.where(output.detach().cpu().numpy()>threshold*0.01,1,0)
            correct = (predicted == answers).sum() #pure accuracy. Get those in the threshold that are correct
            batchTP = np.sum(np.logical_and(predicted == 1, answers== 1))
            batchTN = np.sum(np.logical_and(predicted == 0, answers == 0)) #total TP,TN,FP,FN
            batchFP = np.sum(np.logical_and(predicted == 1, answers == 0)) # does not consider classwise
            batchFN = np.sum(np.logical_and(predicted == 0, answers== 1))
            if threshold not in self._val_accuracies[epochs].keys():
                self._val_accuracies[epochs][threshold] = correct #if not in, append
            else:
                self._val_accuracies[epochs][threshold] = correct + self._val_accuracies[epochs][threshold]
                # else, sum it up.
            if threshold in self._TPlist[epochs].keys():
                # check if this threshold has been recorded for this epoch thus far.
                self._TPlist[epochs][threshold] = self._TPlist[epochs][threshold] + batchTP
                self._FPlist[epochs][threshold] = self._FPlist[epochs][threshold] + batchFP
                self._TNlist[epochs][threshold] = self._TNlist[epochs][threshold] + batchTN 
                # add it to the current count for this epoch.
                self._FNlist[epochs][threshold] = self._FNlist[epochs][threshold] + batchFN 
            else:
                self._TPlist[epochs][threshold] = batchTP
                self._FPlist[epochs][threshold] = batchFP #initiate the count for this epoch
                self._TNlist[epochs][threshold] = batchTN
                self._FNlist[epochs][threshold] = batchFN

            for image_number in range(predicted.shape[0]):
                if threshold not in self._classwise_acc[epochs].keys():
                    self._classwise_acc[epochs][threshold]={}
                    for class_number in range(self._num_classes):
                        self._classwise_acc[epochs][threshold][class_number]={"TN":0, "TP":0,"FN":0,"FP":0}
                for class_number in range(self._num_classes):
                    #reminder classwise_acc format=> classwise_accuracy[epochnumber][threshold][classnumber]
                    if predicted[image_number][class_number]==answers[image_number][class_number]: 
                        # this was a TRUE for this class
                        if answers[image_number][class_number]==0:
                            # TRUE NEGATIVE
                            self._classwise_acc[epochs][threshold][class_number]["TN"] = self._classwise_acc[epochs][threshold][class_number]["TN"] +1
                        else:
                            # TRUE POSITIVE
                            self._classwise_acc[epochs][threshold][class_number]["TP"] = self._classwise_acc[epochs][threshold][class_number]["TP"] +1
                    else:
                        #this was a WRONG prediction for this class
                        if answers[image_number][class_number]==0 and predicted[image_number][class_number]==1:
                            # FALSE NEGATIVE
                            self._classwise_acc[epochs][threshold][class_number]["FP"] = self._classwise_acc[epochs][threshold][class_number]["FP"] +1
                        else:
                            # FALSE POSITIVE
                            self._classwise_acc[epochs][threshold][class_number]["FN"] = self._classwise_acc[epochs][threshold][class_number]["FN"] +1  

