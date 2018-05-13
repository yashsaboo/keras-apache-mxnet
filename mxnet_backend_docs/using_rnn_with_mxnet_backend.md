# Limitations and workaround of RNN layer using MXNet backend

MXNet backend doesn't support Variable length input in Recurrent Layers. However, there is a workaround to this when using MXNet as backend. Below is one of the approach:

### 1. Handling Variable Length Input

Please provide a Fixed length input dimensions(`input_shape`) and `unroll=True` parameters while adding Recurrent layers. Fixed length dimensions mean every sample consist of the same length. Currently, Keras with MXNet backend doesn’t support variable length inputs which means each sample may have a different length. 

**Solution:** You need to pad the input and transform the variable length data to a fixed length data. One of the possible way to do that is as follows:

```python
new_x_train = keras.preprocessing.sequence.pad_sequences(old_x_train, maxlen=MAX_LEN_OF_INPUT_SAMPLE_TYPE_INT)
```

If your input data is of fixed size, you don’t need to pad the input. Keras with MXNet backend requires `input_shape` and `unroll=True` parameters while adding SimpleRNN/LSTM/GRU layer.

`input_shape`: Atleast 2-D of shape `(MAX_LEN_OF_INPUT_SAMPLE_TYPE_INT or time_step, feature_size)`

`unroll`: Boolean. `unroll=True` means it assumes input to be of Fixed Length and `unroll=False` means it assumes input to be of Variable Length.

**Steps(Rough Example):**

1. Transform the input/dataset from variable length to a fixed length(Skip step 1 if you have fixed length input)

```python
new_x_train = keras.preprocessing.sequence.pad_sequences(old_x_train, maxlen=100)
```

2. Build the model by calingl the SimpleRNN/LSTM/GRU layer as shown below:

```python
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)), unroll=True))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

##### len(chars): features size

3. Train the model

```python
model.fit(new_x_train, y_train, batch_size=128, epochs=60)
```

**Note:**

Padding the inputs may cause a performance degradation due to unnecessary computation on Padded data.

### 2. Given input_shape but unroll=False

A user has provided a fixed length inputs which means passing `input_shape` parameters to either RNN layers or an Embedding Layer but forgot to set `unroll=True` while calling RNN layers. 

**Solution:** User needs to provide `unroll=True` while calling RNN layers if they have a fixed length inputs. Please see the above example.