// Connecting to ROS
// -----------------

var ros = new ROSLIB.Ros({
    url : 'ws://' + window.location.hostname + ':9090'
});

ros.on('connection', function() {
    console.log('Client has connected to the server!');
    clear_patch();
});

ros.on('error', function(error) {
    console.log('Error connecting to websocket server: ', error);
});

ros.on('close', function() {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

// Display the original image
var origin_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/raw_img',
    messageType : 'std_msgs/String'
  });

origin_img_listener.subscribe(function(msg) {
    $('#origin').attr("src", "data:image/png;base64," + msg.data);
});

// Display the input image
var input_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/input_img',
    messageType : 'std_msgs/String'
  });

input_img_listener.subscribe(function(msg) {
    $('#input').attr("src", "data:image/png;base64," + msg.data);
});

// Display the perturbation
var perturb_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/perturb_img',
    messageType : 'std_msgs/String'
  });

perturb_img_listener.subscribe(function(msg) {
    $('#patch').attr("src", "data:image/png;base64," + msg.data);
});

// Display the adversarial image
var adv_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/adv_img',
    messageType : 'std_msgs/String'
  });

adv_img_listener.subscribe(function(msg) {
    $('#adv').attr("src", "data:image/png;base64," + msg.data);
});

// Clear Patch
var clear_patch_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/clear_patch',
    messageType : 'std_msgs/Int32'
});

// Fix Patch
var fix_patch_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/fix_patch',
    messageType : 'std_msgs/Int32'
});

// Adversarial Patch Position
var adv_patch_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/adv_patch',
    messageType : 'std_msgs/Int32MultiArray'
});

var boxes = [];

function clear_patch() {
    var clear_patch_msg = new ROSLIB.Message({
        data: 1
    });
    clear_patch_pub.publish(clear_patch_msg);
    boxes = [];
    var ctx=$('#canvas')[0].getContext('2d'); 
    ctx.clearRect(0, 0, 320, 256);
}

function fix_patch(fixed) {
    var fix_patch_msg = new ROSLIB.Message({
        data: parseInt(fixed)
    });
    fix_patch_pub.publish(fix_patch_msg);
}


$(document).ready(function () {

    // Fix patch
    $("#customCheck1").change(function() {
        if(this.checked) {
            fix_patch(1);
            //Do stuff
        }
        else
        {
            fix_patch(0);
        }
    });

    $(function() {
        var ctx=$('#canvas')[0].getContext('2d'); 
        rect = {};
        drag = false;

        move = false;
        box_index = 0;
        startX = 0;
        startY = 0;

        $(document).on('mousedown','#canvas',function(e){
            startX = e.pageX - $(this).offset().left;
            startY = e.pageY - $(this).offset().top;
            console.log(startX, startY);
            for (box_index = 0; box_index < boxes.length; box_index++) {
                if( (startX >= boxes[box_index].startX) && (startY >= boxes[box_index].startY)) {
                    if( ((startX - boxes[box_index].startX) <= boxes[box_index].w) && ((startY - boxes[box_index].startY) <= boxes[box_index].h)) {
                        move = true;
                        break;
                    }
                }
            }
            if(!move){
                rect.startX = e.pageX - $(this).offset().left;
                rect.startY = e.pageY - $(this).offset().top;
                rect.w=0;
                rect.h=0;
                drag = true;
            }
        });
    
        $(document).on('mouseup', '#canvas', function(){
            if(!move){
                drag = false;
                box = [-1, Math.round(rect.startX), Math.round(rect.startY), Math.round(rect.w), Math.round(rect.h)]
                var adv_patch_msg = new ROSLIB.Message({
                    data: box
                });
                adv_patch_pub.publish(adv_patch_msg)
                box = {}
                box.startX = rect.startX
                box.startY = rect.startY
                box.w = rect.w
                box.h = rect.h
                boxes.push(box)
                // console.log(boxes);
            }
            else {
                move = false;
                box = [box_index, Math.round(boxes[box_index].startX), Math.round(boxes[box_index].startY), Math.round(boxes[box_index].w), Math.round(boxes[box_index].h)]
                var adv_patch_msg = new ROSLIB.Message({
                    data: box
                });
                adv_patch_pub.publish(adv_patch_msg)
            }
        });
    
        $(document).on('mousemove',function(e){
            ctx.clearRect(0, 0, 320, 256);
            boxes.forEach(b => {
                ctx.fillRect(b.startX, b.startY, b.w, b.h);
            });
            if (drag) {
                rect.w = (e.pageX - $("#canvas").offset().left)- rect.startX;
                rect.h = (e.pageY - $("#canvas").offset().top)- rect.startY;
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(rect.startX, rect.startY, rect.w, rect.h);
            }
            if (move) {
                boxes[box_index].startX += ((e.pageX - $("#canvas").offset().left) - startX);
                boxes[box_index].startY += ((e.pageY - $("#canvas").offset().top) - startY);
                startX = (e.pageX - $("#canvas").offset().left);
                startY = (e.pageY - $("#canvas").offset().top);
            }
        });    
    });
});
