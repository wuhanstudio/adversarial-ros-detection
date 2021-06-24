// Driving Data from the Server
var steer_data = [];
var adv_data = [];

// Connecting to ROS
// -----------------

var ros = new ROSLIB.Ros({
url : 'ws://localhost:9090'
});

ros.on('connection', function() {
    console.log('Client has connected to the server!');
    var clear_patch_msg = new ROSLIB.Message({
        data: 1
    });
    clear_patch_pub.publish(clear_patch_msg);
    attack(0, $("input[name=flexRadioDefault]:checked").val());
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
// var perturb_img_listener = new ROSLIB.Topic({
//     ros : ros,
//     name : '/perturb_img',
//     messageType : 'std_msgs/String'
//   });

// perturb_img_listener.subscribe(function(msg) {
//     $('#diff').attr("src", "data:image/png;base64," + msg.data);
// });

// Display the adversarial image
var adv_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/adv_img',
    messageType : 'std_msgs/String'
  });

adv_img_listener.subscribe(function(msg) {
    $('#adv').attr("src", "data:image/png;base64," + msg.data);
});

// Activate the attack
var attack_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/attack',
    messageType : 'std_msgs/Int32'
});


// Clear Patch
var clear_patch_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/clear_patch',
    messageType : 'std_msgs/Int32'
});

// Adversarial Patch Position
var adv_patch_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/adv_patch',
    messageType : 'std_msgs/Int32MultiArray'
});

function attack(isAttack, type) {
    if(isAttack)
    {
        var attack_type = 0;
        if(type === 'fgsmr_left')
        {
            attack_type = 1;
        }
        if(type === 'fgsmr_right')
        {
            attack_type = 2;
        }
        var attack_msg = new ROSLIB.Message({
            data: attack_type
        });
        attack_pub.publish(attack_msg);
    }
    else
    {
        var attack_msg = new ROSLIB.Message({
            data: 0
        });
        attack_pub.publish(attack_msg);
    }
}

// Attack Deactivated
function resume() {
    $('#diff').attr("src", "./hold.png");
    $('#adv').attr("src", "./hold.png");
}

var boxes = [];

$(document).ready(function () {

    $(function() {
        var ctx=$('#canvas')[0].getContext('2d'); 
        rect = {};
        drag = false;
    
        $(document).on('mousedown','#canvas',function(e){
            rect.startX = e.pageX - $(this).offset().left;
            rect.startY = e.pageY - $(this).offset().top;
            rect.w=0;
            rect.h=0;
            drag = true;
        });
    
        $(document).on('mouseup',function(){
            drag = false;
            box = [Math.round(rect.startX), Math.round(rect.startY), Math.round(rect.w), Math.round(rect.h)]
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
            console.log(boxes);
        });
    
        $(document).on('mousemove',function(e){
            if (drag) {
                rect.w = (e.pageX - $("#canvas").offset().left)- rect.startX;
                rect.h = (e.pageY - $("#canvas").offset().top)- rect.startY;
                ctx.clearRect(0, 0, 320, 160);
                boxes.forEach(b => {
                    ctx.fillRect(b.startX, b.startY, b.w, b.h);
                });
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(rect.startX, rect.startY, rect.w, rect.h);
            }
        });    
    });
});
