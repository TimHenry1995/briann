'''
class PenStrokeMNIST(Dataset):
    
    def __init__(self, folder_path: str) -> "PenStrokeMNIST":
        # Call super
        super().__init__()

        # Set attributes
        self.folder_path = folder_path
        
    @property
    def folder_path(self) -> str:
        """The path to the folder where the data is stored.

        :return: The path to the folder where the data is stored.
        :rtype: str
        """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, new_value: str) -> None:
        # Input validity
        if not isinstance(new_value, str): raise TypeError(f"The folder_path was expected to be a string but is {type(new_value)}.")
        if not os.path.isdir(new_value): raise ValueError(f"The path {new_value} is not a directory.")
        
        # Set property
        self._folder_path = new_value

    def __len__(self) -> int:
        """The number of instances in the dataset.
        :return: The number of instances in the dataset.
        :rtype: int
        """

        # Get all file names in the data folder
        file_names = os.listdir(self.folder_path)

        # Filter for files that start with 'trainimg-' or 'testimg-'
        instance_count = 0
        for file_name in file_names:
            if 'inputdata' in file_name: instance_count += 1
        
        # Output
        return instance_count
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        file_path = os.path.join(self.folder_path, f"img-{index}-targetdata.txt" )
        
        with open(file=file_path, mode='r') as file_handle:
            text_data = file_handle.read()

        # Parse
        # The first 10 columns one-hot encode the label. All but the first rows can be ignored.
        # The next 4 columns encode the dx, dy, end of stroke (0,1) and end of sequence (0, 1). The end of sequence can be ignored
        lines = text_data.split('\n')
        y = torch.Tensor([(float)(entry) for entry in lines[0].split(' ')[:10]])
        
        x = [None] * len(lines)
        for l, line in enumerate(lines):
            if len(line) > 0: # Exclude possible empty lines
                x[l] = [(float)(entry) for entry in line.split(' ')[10:13]]
        del x[l:]
        
        x = torch.Tensor(x)

        return x,y

if __name__=="__main__":
    path = bpufm.map_path_to_os(path=os.path.join("tests","data","Pen Stroke MNIST"))
    dataset = PenStrokeMNIST(folder_path=path)
    #dataset = Sinusoids(instance_count=10, duration=3.0, sampling_rate=50, frequency_count=2, noise_range=0.02)
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=lambda sequences: collate_function(sequences=sequences, batch_first=True))
    
    for X, y in data_loader:
        print(X.shape, y.shape)
        '''