# TODO List

## Right Now

- [ ] Adapt tokenizer from huggingface's transformers tokenizer to tensorflow.

## High Priority

- [ ] **Model Training Pipeline**: Implement the pipeline for training the Transformer model using the provided data and preprocessing steps.
- [ ] **Hyperparameter Tuning**: Experiment with different hyperparameters to optimize the model's performance on the specific task.
- [ ] **Evaluation Metrics**: Define evaluation metrics (e.g., BLEU score, perplexity) to assess the model's effectiveness and generalization.

## Medium Priority

- [ ] **Inference Script**: Develop a script for making predictions using a trained Transformer model. Ensure the script handles input sequences and generates coherent output sequences.
- [ ] **Transfer Learning**: Investigate the feasibility of using pre-trained Transformer models (e.g., BERT, GPT) as encoder or decoder components to leverage their contextual understanding.

## Low Priority

- [ ] **Custom Dataset Support**: Implement a data loading pipeline that can handle custom datasets in various formats (e.g., JSON, XML) for greater flexibility. (KerasNLP dataset im looking at you)
- [ ] **Model Serialization**: Integrate functionality for saving and loading trained models, allowing seamless deployment and further experimentation.
- [ ] **Regularization Strategies**: Explore additional regularization techniques (e.g., layer normalization, weight decay) to enhance model generalization and stability.

## Documentation

- [ ] **Code Comments**: Add detailed comments to the code to explain complex sections, functions, and classes for future reference.
- [ ] **Usage Instructions**: Expand on the provided documentation to guide users on how to adapt and use the Transformer model for their specific tasks.
- [ ] **Sample Data**: Include a sample CSV file and instructions for users to easily understand and test the code.

## Future Enhancements

- [ ] **Attention Visualization**: Develop interactive tools for visualizing attention maps, allowing users to interact with the model's attention mechanisms.

---
Last Updated: 2023-08-23
