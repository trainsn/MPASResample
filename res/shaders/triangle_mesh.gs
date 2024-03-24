#version 430 core 
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;
//layout (points, max_vertices = 3) out;

in V2G{
	flat int vertex_id;
}v2g[];

out float CLIMATE_VALS_VAR;

//connectivity
uniform samplerBuffer latCell;
uniform samplerBuffer lonCell;
uniform isamplerBuffer cellsOnVertex;
uniform samplerBuffer temperature;

uniform int TOTAL_LAYERS;
uniform int layer_id;

const float M_PI = 3.14159265358;

void main(){
	// only consider outter most surface mesh 
	int vertexId = v2g[0].vertex_id;
	// query three nerighboring cell id.
	ivec3 cellId3 = texelFetch(cellsOnVertex, vertexId - 1).xyz;
	if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){
	}
	else {
		int cellIds[3];
	
		cellIds[0] = cellId3.x;
		cellIds[1] = cellId3.y;
		cellIds[2] = cellId3.z;

		float lat[3];
		float lon[3];
		bool showTriangle = true; 
		for (int i = 0; i < 3; i++){
			int cell_id = cellIds[i];
			lat[i] = texelFetch(latCell, cell_id - 1).x;
			lon[i] = texelFetch(lonCell, cell_id - 1).x; 
			if (abs(lat[i]) > M_PI * 80.0 / 90.0 / 2.0){		// only show tropical areas 
				showTriangle = false;
				break;
			}
		}
		if (showTriangle){	// filter out triangle cross Prime meridian
			showTriangle = abs(lon[0] - lon[1]) < M_PI / 2 && abs(lon[0] - lon[2]) < M_PI / 2 && abs(lon[1] - lon[2]) < M_PI / 2;
		}
		if (showTriangle){
			for (int i = 0; i < 3; i++){
				int cell_id = cellIds[i];
				vec4 xyzw = vec4(
								(lon[i] - M_PI) / M_PI,
								lat[i] * 2 / M_PI,
								0.0,
								1.0);

				gl_Position = xyzw;
				CLIMATE_VALS_VAR = texelFetch(temperature, (cell_id - 1) * TOTAL_LAYERS + layer_id).x;
				EmitVertex();
			}	
		}
	}
	EndPrimitive();
}