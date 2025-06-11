using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Udp;

public class GrabObjects : MonoBehaviour
{
    [SerializeField] private Transform grab_point;
    [SerializeField] private Transform ray_point;
    [SerializeField] private float ray_distance = 1.5f;
    [SerializeField] private bool UDP_Connection = false;
    [SerializeField] public bool GameMode2 = false;

    public UdpHost UdpController;
    public TargetCheck target_controller;
    private GameObject grabbed_obj;
    private int layer_idx;
    [HideInInspector] public bool release;

    private void Start()
    {
        layer_idx = LayerMask.NameToLayer("Objects");
    }

    private void OnEnable()
    {
        UdpHost.OnReceiveMsg += OnBCIMessage;
    }

    private void OnDisable()
    {
        UdpHost.OnReceiveMsg -= OnBCIMessage;
    }

    private void OnBCIMessage(string msg)
    {
        msg = msg.Trim().ToLower();

        if (msg == "grab" && grabbed_obj == null)
        {
            Debug.Log("BCI GRAB received – attempting to grab object.");
            TryGrabFromRaycast();
        }
    }


    void Update()
    {

        if (release && grabbed_obj != null)
        {
            ReleaseObj();
        }
    }
    void TryGrabFromRaycast()
    {
        RaycastHit2D hit_info = Physics2D.Raycast(ray_point.position, Vector2.down, ray_distance);
        Debug.DrawRay(ray_point.position, Vector2.down * ray_distance, Color.red, 1f);

        if (hit_info.collider != null)
        {
            Debug.Log("✅ Raycast hit: " + hit_info.collider.name);

            GameObject candidate = hit_info.collider.gameObject;

            if (candidate.layer == layer_idx)
            {
                if (target_controller.CheckObj(candidate))
                {
                    grabbed_obj = candidate;

                    var rb = grabbed_obj.GetComponent<Rigidbody2D>();
                    if (rb != null)
                        rb.isKinematic = true;

                    grabbed_obj.transform.SetParent(grab_point);
                    grabbed_obj.transform.localPosition = Vector3.zero;

                    if (UDP_Connection)
                        UdpController.SendMsg(GameMode2 ? "mode2" : "up");

                    Debug.Log("✅ Flower grabbed: " + grabbed_obj.name);
                }
                else
                {
                    Debug.Log("❌ Flower not part of sequence: " + candidate.name);
                }
            }
            else
            {
                Debug.Log("⚠️ Ray hit an object not on the 'Objects' layer: " + candidate.name);
            }
        }
        else
        {
            Debug.Log("❌ Raycast did not hit anything.");
        }
    }

    

    public void GrabObjectIfValid()
    {
        if (grabbed_obj == null)
        {
            TryGrabFromRaycast();
        }
    }

    public void ReleaseObj()
    {
        if (grabbed_obj != null)
        {
            if (UDP_Connection)
                UdpController.SendMsg("down");

            // Detach so it stops following hand
            grabbed_obj.transform.SetParent(null);

            // Target-based snap placement
            target_controller.Place(grabbed_obj);

            // Cleanup
            grabbed_obj = null;
            release = false;
        }
    }
    public GameObject GetHeldFlower()
    {
        return grabbed_obj;
    }

}
