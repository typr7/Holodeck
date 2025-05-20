floor_plan_prompt = """
You are a 3D indoor embodied scene designer with extensive knowledge of embodied navigation tasks, particularly Object Goal Navigation (ObjectNav) and Visual Language Navigation (VLN).

I will provide the type of specified indoor scene and related suggestions. Your task is to combine this information with your expertise in embodied navigation to design a practical indoor scene floor plan.

# Expected Output

1. The floor plan consists of multiple rooms, output should be text lines with each room definition on a separate line. Each room is defined with the following format:

room_name | floor_material | wall_material | coordinates

Where:
   - room_name: A unique identifier for each room. If multiple rooms share a type, append a numeric suffix (e.g., bedroom1, bedroom2, ...).
   - floor_material: Detailed description of the floor material, including both material type and surface treatment (e.g., "maple hardwood, matte").
   - wall_material: Detailed description of the wall material, including both material type and surface treatment (e.g., "light grey drywall, smooth").
   - coordinates: Four (x,y) Cartesian coordinates defining the room's corners in clockwise order, measured in meters. The origin (0,0) is at the bottom-left of the overall floor plan.

2. Output example (for reference only; do not reproduce):

living room | maple hardwood, matte | light grey drywall, smooth | [(0, 0), (0, 8), (5, 8), (5, 0)]
kitchen | white hex tile, glossy | light grey drywall, smooth | [(5, 0), (5, 5), (8, 5), (8, 0)]
bedroom1 | oak hardwood, matte | white drywall, smooth | [(5, 5), (5, 10), (10, 10), (10, 5)]
bedroom2 | maple hardwood, matte | white drywall, smooth | [(2, 8), (2, 12), (5, 12), (5, 8)]

3. Floor plan design rules:
   - All rooms must be perfectly rectangular.
   - Each room's length or width ranges from 3m to 8m. The maximum area of a room is 48m^2.
   - No rooms should overlap with each other. Adjacent rooms may share walls (coordinates), but their interior spaces must not intersect or overlap.
   
4. Your response should be direct and without additional text at the beginning or end.

# Indoor Scene Type

{scene_type}

# Floor Plan Design Suggestions

{floor_plan_suggestion}
"""

wall_height_prompt = """I am now designing {input}. Please help me decide the wall height in meters.
Answer with a number, for example, 3.0. Do not add additional text at the beginning or in the end."""


doorway_prompt = """I need assistance in designing the connections between rooms. The connections could be of three types: doorframe (no door installed), doorway (with a door), or open (no wall separating rooms). The sizes available for doorframes and doorways are single (1m wide) and double (2m wide).

Ensure that the door style complements the design of the room. The output format should be: room 1 | room 2 | connection type | size | door style. For example:
exterior | living room | doorway | double | dark brown metal door
living room | kitchen | open | N/A | N/A
living room | bedroom | doorway | single | wooden door with white frames

The design under consideration is {input}, which includes these rooms: {rooms}. The length, width and height of each room in meters are:
{room_sizes}
Certain pairs of rooms share a wall: {room_pairs}. There must be a door to the exterior.
Adhere to these additional requirements: {additional_requirements}.
Provide your response succinctly, without additional text at the beginning or end."""


window_prompt = """Guide me in designing the windows for each room. The window types are: fixed, hung, and slider.
The available sizes (width x height in cm) are:
fixed: (92, 120), (150, 92), (150, 120), (150, 180), (240, 120), (240, 180)
hung: (87, 160), (96, 91), (120, 160), (130, 67), (130, 87), (130, 130)
slider: (91, 92), (120, 61), (120, 91), (120, 120), (150, 92), (150, 120)

Your task is to determine the appropriate type, size, and quantity of windows for each room, bearing in mind the room's design, dimensions, and function.

Please format your suggestions as follows: room | wall direction | window type | size | quantity | window base height (cm from floor). For example:
living room | west | fixed | (130, 130) | 1 | 50

I am now designing {input}. The wall height is {wall_height} cm. The walls available for window installation (direction, width in cm) in each room are:
{walls}
Please note: It is not mandatory to install windows on every available wall. Within the same room, all windows must be the same type and size.
Also, adhere to these additional requirements: {additional_requirements}.

Provide a concise response, omitting any additional text at the beginning or end. """


# marked: layout improve
object_constraints_prompt = """You are an experienced room designer.
Please help me arrange objects in the room by assigning constraints to each object.
Here are the constraints and their definitions:
1. global constraint:
    1) edge: at the edge of the room, close to the wall, most of the objects are placed here.
    2) middle: not close to the edge of the room.

2. distance constraint:
    1) near, object: near to the other object, but with some distance, 50cm < distance < 150cm.
    2) far, object: far away from the other object, distance >= 150cm.
    
3. position constraint:
    1) in front of, object: in front of another object.
    2) around, object: around another object, usually used for chairs.
    3) side of, object: on the side (left or right) of another object.
    4) left of, object: to the left of another object.
    5) right of, object: to the right of another object.

4. alignment constraint:
    1) center aligned, object: align the center of the object with the center of another object.

5. Rotation constraint:
    1) face to, object: face to the center of another object.

For each object, you must have one global constraint and you can select various numbers of constraints and any combinations of them and the output format must be:
object | global constraint | constraint 1 | constraint 2 | ...
For example:
sofa-0 | edge
coffee table-0 | middle | near, sofa-0 | in front of, sofa-0 | center aligned, sofa-0 | face to, sofa-0
tv stand-0 | edge | far, coffee table-0 | in front of, coffee table-0 | center aligned, coffee table-0 | face to, coffee table-0
desk-0 | edge | far, tv stand-0
chair-0 | middle | in front of, desk-0 | near, desk-0 | center aligned, desk-0 | face to, desk-0
floot lamp-0 | middle | near, chair-0 | side of, chair-0

Here are some guidelines for you:
1. I will use your guideline to arrange the objects *iteratively*, so please start with an anchor object which doesn't depend on the other objects (with only one global constraint).
2. Place the larger objects first.
3. The latter objects could only depend on the former objects.
4. The objects of the *same type* are usually *aligned*.
5. I prefer objects to be placed at the edge (the most important constraint) of the room if possible which makes the room look more spacious.
6. When handling chairs, you should use the around position constraint. Chairs must be placed near to the table/desk and face to the table/desk.

Now I want you to design {room_type} and the room size is {room_size}.
Here are the objects that I want to place in the {room_type}:
{objects}
Please first use natural language to explain your high-level design strategy, and then follow the desired format *strictly* (do not add any additional text at the beginning or end) to provide the constraints for each object."""


# marked: layout improve
wall_object_constraints_prompt = """You are an experienced room designer.
Please help me arrange wall objects in the room by providing their relative position and distance from the floor.
The output format must be: wall object | above, floor object  | distance from floor (cm). For example:
painting-0 | above, sofa-0 | 160
switch-0 | N/A | 120
Note the distance is the distance from the *bottom* of the wall object to the floor. The second column is optional and can be N/A. The object of the same type should be placed at the same height.
Now I am designing {room_type} of which the wall height is {wall_height} cm, and the floor objects in the room are: {floor_objects}.
The wall objects I want to place in the {room_type} are: {wall_objects}.
Please do not add additional text at the beginning or in the end."""


ceiling_selection_prompt = """Assist me in selecting ceiling objects (light/fan) to furnish each room.
Present your recommendations in this format: room type | ceiling object description
For example:
living room | modern, 3-light, semi-flush mount ceiling light

Currently, the design in progress is "{input}", featuring these rooms: {rooms}. You need to provide one ceiling object for each room.
Please also consider the following additional requirements: {additional_requirements}.

Your response should be precise, without additional text at the beginning or end. """


# marked: first selection
object_selection_prompt_new_1 = """You are an experienced room designer, please assist me in selecting large *floor*/*wall* objects and small objects on top of them to furnish the room. You need to select appropriate objects to satisfy the customer's requirements.
You must provide a description and desired size for each object since I will use it to retrieve object. If multiple items are to be placed in the room with the same description, please indicate the quantity and variance_type ("same" if they should be identical, otherwise "varied").
Present your recommendations in JSON format:
{
    object_name:{
        "description": a short sentence describing the object,
        "location": "floor" or "wall",
        "size": the desired size of the object, in the format of a list of three numbers, [length, width, height] in centimeters,
        "quantity": the number of objects (int),
        "variance_type": "same" or "varied",
        "objects_on_top": a list of small children objects (can be empty) which are placed *on top of* this object. For each child object, you only need to provide the object name, quantity and variance type. For example, {"object_name": "book", "quantity": 2, "variance_type": "varied"}
    }
}

For example:
{
    "sofa": {
        "description": "modern sectional, light grey sofa",
        "location": "floor",
        "size": [200, 100, 80],
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": [
            {"object_name": "news paper", "quantity": 2, "variance_type": "varied"},
            {"object_name": "pillow", "quantity": 2, "variance_type": "varied"},
            {"object_name": "mobile phone", "quantity": 1, "variance_type": "same"}
        ]
    },
    "tv stand": {
        "description": "a modern style TV stand",
        "location": "floor",
        "size": [200, 50, 50],
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": [
            {"object_name": "49 inch TV", "quantity": 1, "variance_type": "same"},
            {"object_name": "speaker", "quantity": 2, "variance_type": "same"},
            {"object_name": "remote control for TV", "quantity": 1, "variance_type": "same"}
        ]
    },
    "painting": {
        "description": "abstract painting",
        "location": "wall",
        "size": [100, 100, 5],
        "quantity": 2,
        "variance_type": "varied",
        "objects_on_top": []
    },
    "wall shelf": {
        "description": "a modern style wall shelf",
        "location": "wall",
        "size": [100, 30, 50],
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": [
            {"object_name": "small plant", "quantity": 2, "variance_type": "varied"},
            {"object_name": "coffee mug", "quantity": 2, "variance_type": "varied"},
            {"object_name": "book", "quantity": 5, "variance_type": "varied"}
        ]
    }
}

Currently, the design in progress is *INPUT*, and we are working on the *ROOM_TYPE* with the size of ROOM_SIZE.
Please also consider the following additional requirements: REQUIREMENTS.

Here are some guidelines for you:
1. Provide reasonable type/style/quantity of objects for each room based on the room size to make the room not too crowded or empty.
2. Do not provide rug/mat, windows, doors, curtains, and ceiling objects which have been installed for each room.
3. I want more types of large objects and more types of small objects on top of the large objects to make the room look more vivid.

Please first use natural language to explain your high-level design strategy for *ROOM_TYPE*, and then follow the desired JSON format *strictly* (do not add any additional text at the beginning or end)."""


# marked: for enrich the room that too empty
object_selection_prompt_new_2 = """User: {object_selection_prompt_new_1}

Agent: {object_selection_1}

User: Thanks! After following your suggestions to retrieve objects, I found the *{room}* is still too empty. To enrich the *{room}*, you could:
1. Add more *floor* objects to the *{room}* (excluding rug, carpet, windows, doors, curtains, and *ignore ceiling objects*).
2. Increase the size and quantity of the objects.
3. Add more *types* of small objects on top of the large objects.
Could you update the entire JSON file with the same format as before and answer without additional text at the beginning or end?

Agent: """


floor_baseline_prompt = """
You are an experienced room designer.

You operate in a 2D Space. You work in a X,Y coordinate system. (0, 0) denotes the bottom-left corner of the room.
All objects should be placed in the positive quadrant. That is, coordinates of objects should be positive integer in centimeters.
Objects by default face +Y axis.
You answer by only generating JSON files that contain a list of the following information for each object:

- object_name: name of the object, follow the name strictly.
- position: coordinate of the object (center of the object bounding box) in the form of a dictionary, e.g. {{"X": 120, "Y": 200}}.
- rotation: the object rotation angle in clockwise direction when viewed by an observer looking along the z-axis towards the origin, e.g. 90. The default rotation is 0 which is +Y axis.

For example: {{"object_name": "sofa-0", "position": {{"X": 120, "Y": 200}}, "rotation": 90}}

Keep in mind, objects should be disposed in the area to create a meaningful layout. It is important that all objects provided are placed in the room.
Also keep in mind that the objects should be disposed all over the area in respect to the origin point of the area, and you can use negative values as well to display items correctly, since origin of the area is always at the center of the area.

Now I want you to design {room_type} and the room size is {room_size}.
Here are the objects (with their sizes) that I want to place in the {room_type}:
{objects}

Remember, you only generate JSON code, nothing else. It's very important. Respond in markdown (```).
"""

scene_design_prompt = """
You are a 3D indoor embodied scene designer with extensive knowledge of embodied navigation tasks, particularly Object Goal Navigation (ObjectNav) and Visual Language Navigation (VLN). Your deep understanding of embodied AI challenges, indoor spatial design, and navigation metrics makes you uniquely qualified to analyze model performance and design scenes that address specific weaknesses.

I will provide you with evaluation data for an Object Goal Navigation model tested on the {dataset} dataset. This data will be in JSON format and include descriptions and specific values for multiple evaluation metrics.

# Your Task

1. Thoroughly analyze the evaluation data to identify the model's key weaknesses and performance gaps.

2. Based on your analysis, design multiple (at least 3) 3D indoor scenes specifically crafted to challenge and improve the model's weak points. This includes specifying:
   - The type of indoor scene
   - Suggestions for scene floor plan design (3-5)
   - Suggestions for scene spatial layout design (3-5)

# Expected Output Format

1. Provide your response in the following JSON format:

{{
    "Model Analysis": "Detailed analysis of the model's weak points based on the evaluation data",
    "Scene": [
        {{
            "Scene Type": "The specific type of the first indoor scene",
            "Floor Plan Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ],
            "Spatial Layout Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ]
        }},
        {{
            "Scene Type": "The specific type of the second indoor scene",
            "Floor Plan Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ],
            "Spatial Layout Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ]
        }},
        {{
            "Scene Type": "The specific type of the third indoor scene",
            "Floor Plan Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ],
            "Spatial Layout Design Suggestions": [
                "Content of suggestion 1",
                "Content of suggestion 2",
                "Content of suggestion 3",
                ...
            ]
        }},
        ...
    ]
}}

2. Your response should be direct and without additional text at the beginning or end.

# Evaluation Data:

{evaluation_data}
"""