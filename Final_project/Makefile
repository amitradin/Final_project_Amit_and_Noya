CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
OBJS = symnmf.o matrix_ops.o

symnmf: $(OBJS)
	$(CC) $(CFLAGS) -o symnmf $(OBJS) -lm

symnmf.o: symnmf.c symnmf.h matrix_ops.h
	$(CC) $(CFLAGS) -c symnmf.c

matrix_ops.o: matrix_ops.c matrix_ops.h
	$(CC) $(CFLAGS) -c matrix_ops.c

clean:
	rm -f symnmf $(OBJS)

.PHONY: clean
