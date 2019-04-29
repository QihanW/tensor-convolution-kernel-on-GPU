FILE1 = con_den_1.cu
FILE2 = con_den_2.cu
FILE3 = con_spa.cu
FILE4 = con_filter_spa.cu

TARGET1 = con_den_1
TARGET2 = con_den_2
TARGET3 = con_spa
TARGET3 = filter_spa

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): $(FILE1)
        nvcc $(FILE1) -o $(TARGET1)

$(TARGET2): $(FILE2)
        nvcc $(FILE2) -o $(TARGET2)

$(TARGET3): $(FILE3)
        nvcc $(FILE3) -o $(TARGET3)
        
$(TARGET4): $(FILE4)
        nvcc $(FILE4) -o $(TARGET4)

.PHONY : clean
clean :
        rm $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)
