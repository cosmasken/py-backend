# My AI Project

This project is designed to train a small AI model using images and their annotations in Pascal VOC format. The goal is to provide a straightforward implementation that can be used as a foundation for further development and experimentation.

## Project Structure

```
my-ai-project
├── data
│   ├── images          # Directory containing image files for training
│   └── annotations     # Directory containing Pascal VOC format annotation files
├── src
│   ├── train.py        # Script for training the AI model
│   ├── model.py        # Defines the architecture of the AI model
│   └── utils.py        # Utility functions for data handling and preprocessing
├── requirements.txt     # Lists the required Python dependencies
└── README.md            # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/cosmasken/py-backend
   cd py-backend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your images in the `data/images` directory.
   - Place your Pascal VOC annotations in the `data/annotations` directory.

## Usage

To train the AI model, run the following command:
```
python src/train.py
```

This will load the images and annotations, preprocess the data, and start the training process.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.