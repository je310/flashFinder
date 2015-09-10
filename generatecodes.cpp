#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using namespace std;


template<unsigned int N>
vector<int> corrCode(bitset<N> bits){
    vector<int> input(bits.size());
    vector<int> output(input.size(), 0);

    for(size_t i = 0; i != input.size(); i++)
        input[i] = bits[i];

    for(size_t i = 0; i != input.size(); i++){
        for(size_t j = 0; j!= input.size(); j++){
            output[i] += input[j]*input[(j+i)%input.size()];
        }
    }

	int min = *(min_element(output.begin(), output.end())); 

	for(size_t i = 0; i != output.size(); i++)
		output[i] -= min;

    return output;
}

int main(){
	const unsigned int codesize = 8;
	
	set< vector<int> > foundcodes;
 
	for(unsigned long i = 1; i < pow(2,codesize) - 1; i++){
		bitset<codesize> code = i;
	
		vector<int> ccode = corrCode<codesize>(code);

		if  (foundcodes.find(ccode) == foundcodes.end()){
			cout << i << ":\t";
			for (size_t x = 0; x!= code.size(); x++)
				cout << code[x] << " ";
			cout << "\t";
			for (size_t x = 0; x!= ccode.size(); x++)
				cout << ccode[x] << " ";
			cout << endl;
			foundcodes.insert(ccode);
		}
	}

	cout << "Number of codes: " << foundcodes.size() << endl;

	return 0;
}

