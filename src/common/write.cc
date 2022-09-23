#include <stdio.h>
#include <cstdlib>
#include <malloc.h>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
class data
{
public:
	size_t nptr = 0;
	size_t *sizes;
	void save(int fd)
	{
		size_t totsize = 0;
		sizes = (size_t *)malloc(sizeof(size_t) * nptr);
        write(fd, getPtr(), getSize());
        totsize+=getSize();
		for (int i = 0; i < nptr; i++)
		{
			sizes[i] = malloc_usable_size(getMemberPointer(i));
		}
		write(fd, sizes, sizeof(size_t) * nptr);
		totsize+=sizeof(size_t) * nptr;
		for (int i = 0; i < nptr; i++)
		{
			write(fd, getMemberPointer(i), malloc_usable_size(getMemberPointer(i)));
			totsize+=malloc_usable_size(getMemberPointer(i));
		}
		printf("saved total size=%ld\n",totsize);
	}
	void readfile(int fd)
	{
		sizes = (size_t *)malloc(sizeof(size_t) * nptr);
		read(fd, sizes, sizeof(size_t) * nptr);
		for (int i = 0; i < nptr; i++)
		{
			getMemberPointer(i) = malloc(sizes[i]);
			read(fd, getMemberPointer(i), sizes[i]);
		}
	}
	virtual void *&getMemberPointer(int i) {return getMemberPointer(i);};
    virtual void *getPtr(){return this;};
    virtual size_t getSize(){return sizeof(this);};
	data() {nptr = 0;}
};
template <class T>
T *readdata(int fd, T *&ptr)
{
	ptr = (T *)malloc(sizeof(T));
	read(fd, ptr, sizeof(T));
	int fd1 = open("test1.txt", O_RDWR, S_IRUSR, S_IWUSR);
	write(fd1, ptr, sizeof(T));
	close(fd1);
	printf("got nptr=%ld, npptr=%ld\n",ptr->nptr,ptr->parentnptr);
	ptr->readfile(fd);
	return ptr;
}

class d1 : public data
{
public:
	size_t parentnptr=0;
	d1() : data(), parentnptr(nptr) { nptr += member_pointers.size(); }
	union
	{
		struct
		{
			int *a;
			double *b;
		} mp;
		std::array<void *, sizeof(mp) / sizeof(void *)> member_pointers;
	};
	void *&getMemberPointer(int i)
	{
		//if(i<parentnptr) return data::getMemberPointer(i);
		return member_pointers[i];
	}
    virtual void *getPtr(){return this;};
    virtual size_t getSize(){return sizeof(d1);};
};

int main(void)
{
	// create a class and write the class to a file.
	int fd = open("test.txt", O_RDWR, S_IRUSR, S_IWUSR);
	int n = 1;
	d1 *md = new d1();
	md->mp.a = (int *)malloc(sizeof(int) * n);
	md->mp.b = new double[n];
	for (int i = 0; i < n; i++)
	{
		md->mp.a[i] = 'a' + i;
		md->mp.b[i] = sin(i+1);
	}

	md->save(fd);
	close(fd);

	// read a class from a file.
	fd = open("test.txt", O_RDONLY, S_IRUSR, S_IWUSR);
	d1 *mdr;
	readdata(fd, mdr);
	//for (int i = 0; i < 100000; i++)
	//{
	//	printf("new data: %d, %f\n", mdr->mp.a[i],mdr->mp.b[i]);
	//}
	close(fd);
}
