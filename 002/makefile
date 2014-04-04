
all: 
	if [ ! -d build ]; then mkdir build; fi
	cd build; cmake .. ; make
	
run: all
	cd build; ./main

clean: 
	make -C build clean
	
	
# try to remove all
clean-all: 
	rm -rf build