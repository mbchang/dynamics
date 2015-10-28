CNN
      Conv Layer 1 input: (150, 150)
      Conv Layer 1 output: (74, 74)

      Conv Layer 2 input: (74, 74)
      Conv Layer 2 output: (36, 36)

      Conv Layer 3 input: (36, 36)
      Conv Layer 3 output: (17, 17)

      FC Layer input: 24 * 17 * 17
      FC Layer output: 100

LSTM 
      Input Dim: 100
      Hidden Dim: 50  
      Output Dim: 100

DCNN
      FC Layer input: 100
      FC Layer output: 24 * 17 * 17

      DeConv Layer 1 input: (17, 17)
      DeConv Layer 1 output: (36, 36)

      DeConv Layer 2 input: (36, 36)
      DeConv Layer 2 output: (74, 74)

      DeConv Layer 3 input: (74, 74)
      DeConv Layer 3 output: (150, 150)


Notes
    Not sure if the .t7 file in 'physics_engine' is needed
