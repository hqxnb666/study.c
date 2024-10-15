#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#define PPP 0x9e3779b9
#define max ((( t>>5 ^ i << 2)+(i>>3^t<<4))^((result^i)+(flag[(p&3)^e]^t)))

void publc(uint32_t* option, int num, uint32_t const flag[4])
{
	uint32_t i;
	uint32_t t;
	uint32_t result;
	unsigned p;
	unsigned yuan;
	unsigned e;
	if (num > 1)
	{
		yuan = 415 / num;
		yuan += 114;
		result = 0;
		t = option[yuan - 1];
		while (yuan)
		{
			result += PPP;
			e = (result >> 2) & 3;
			for (int p = 0; p < num - 1; p++)
			{
				i = option[p + 1];
				t = option[p];
				t += max;
			}
			i = option[0];
			t = option[num - 1];
			t += max;
			yuan--;
		}
	
	}
	else if (num < -1)
	{
		num = num * -1;
		yuan = 415 / num + 114;
		result = yuan * PPP;
		i = option[0];
		while (yuan)
		{
			e = (result >> 2) & 3;
			for (int p = num - 1; p > 0; p--)
			{
				t = option[p - 1];
				i = option[p];
				i -= max;
			}
			t = option[num - 1];
			i = option[0];
			i -= max;
			result -= PPP;
			yuan--;
		}
	}
}
uint32_t twictint(char* hex)
{
	uint32_t zhi = 0;
	int zhi1 = 0;
	while (*hex && zhi1 < 8)
	{
		uint8_t bite = *hex++;
		if (bite >= '0' && bite <= '9') {
			bite = bite - '0';
		}
		else if (bite >= 'a' && bite <= 'f') {
			bite = bite - 'a' + 10;
		}
		else if(bite >= 'A' && bite <= 'F'){
			bite = bite - 'A' + 10;
		}
		zhi = (zhi << 4) | (bite & 0xF);
		zhi1++;
	}
	return zhi;
}
int main() {
	uint32_t option1[0x10] = { 0 };
	uint32_t option2[4] = { 0 };
	option1[0] = 0x79696755;
	option1[1] = 0x67346F6C;
	option1[2] = 0x69231231;
	option1[3] = 0x5F674231;
	option2[0] = 0x480AC20C;
	option2[1] = 0xCE9037F2;
	option2[2] = 0x8C212018;
	option2[3] = 0xE92A18D;
	option2[4] = 0xA4035274;
	option2[5] = 0x2473AAB1;
	option2[6] = 0xA9EFDB58;
	option2[7] = 0xA52CC5C8;
	option2[8] = 0xE432CB51;
	option2[9] = 0xD04E9223;
	option2[10] = 0x6FD07093;

	publc(option2, -11, option1);
	for (int i = 0; i < 11 * 4; i++) {
		std::cout << *(((unsigned char*)option2) + i);
	}
	std::cout << std::endl;
}