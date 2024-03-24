#version 430 core 

in float CLIMATE_VALS_VAR;
out float out_Color;
uniform float tMin;
uniform float tMax;

void main(){
	float scalar = (CLIMATE_VALS_VAR - tMin) / (tMax - tMin);
	out_Color = scalar;
}