using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Udp;
using System;

public class PlayerMovement : MonoBehaviour
{
    public CharacterController2D controller;
    public GrabObjects obj_controller;
    public float run_speed = 40f;

    public Transform thresholdTransform;  // Assign this in Inspector (drag ThresholdLine here)

    private bool lift = false;
    private bool allowBCI = false;
    private bool hasLifted = false;

    private Vector2 handStartPos;
    private bool returning = false;

    [HideInInspector] public bool release = false;

    private GameObject heldFlower = null;
    public Transform handTransform; // Assign in inspector

    void Start()
    {
        allowBCI = false;
        handStartPos = controller.m_Rigidbody2D.position;
    }

    void Update()
    {
        if (Input.GetKeyDown("space")) {
            allowBCI = true;
            Debug.Log("BCI control enabled.");
        }


        if (Input.GetKeyDown(KeyCode.Escape))
        {
            lift = false;
            returning = true;
            hasLifted = false;
            Debug.Log("❌ Lift manually canceled.");
        }

        MovementsController();
    }

    void FixedUpdate()
    {
        float y = controller.m_Rigidbody2D.position.y;
        float thresholdY = thresholdTransform.position.y;

        // ➤ Lift in normal mode
        if (lift)
        {
            if (y < thresholdY - 0.01f) // small tolerance to ensure lift happens
            {
                controller.LiftObject();
                Debug.Log(thresholdY);
                hasLifted = true;
            }
            else if (hasLifted)
            {
                lift = false;
                returning = true;
                hasLifted = false;

                if (heldFlower != null)
                {
                    heldFlower.transform.SetParent(null);
                    heldFlower = null;
                }

                obj_controller.ReleaseObj();
                Debug.Log("Flower released (normal mode)");
            }
        }

        // ➤ Return hand downward after lift

        else if (returning)
        {
            controller.m_Rigidbody2D.position = handStartPos; // Snap back to start position
            controller.m_Rigidbody2D.velocity = Vector2.zero; // Stop any movement
            returning = false;
            lift = false;
            hasLifted = false;

            Debug.Log("Hand returned to start position");
        }
        /*else if (!returning && y <= handStartPos.y) // If not returning and at start position
        {
            returning = false;
            Debug.Log("Hand returned to start position.");
        }*/
    }

    void MovementsController()
    {
        if (Input.GetKeyDown("left")) controller.MoveTile('L');
        else if (Input.GetKeyDown("right")) controller.MoveTile('R');
        else if (Input.GetKeyDown("up")) controller.MoveTile('U');
        else if (Input.GetKeyDown("down")) controller.MoveTile('D');

        /*else if (Input.GetKeyDown("space"))
        {
            lift = true;
        }*/
    }

    void OnEnable()
    {
        UdpHost.OnReceiveMsg += OnBCIMessage;
    }

    void OnDisable()
    {
        UdpHost.OnReceiveMsg -= OnBCIMessage;
    }

    void OnBCIMessage(string msg)
    {

        msg = msg.Trim().ToLower();

        if (msg == "grab")
        {   
            allowBCI = true;
            Debug.Log("GRAB detected, flower picked.");
        }
        if (msg == "lift" && allowBCI)
        {
            lift = true;
            Debug.Log("RIGHT ARM detected – triggering lift.");
        }
    }
}
