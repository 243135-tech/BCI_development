using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Udp;

public class TargetCheck : MonoBehaviour
{
    [SerializeField] private GameObject floating_text;
    public List<GameObject> targets = new List<GameObject>();

    private List<string> target_list = new List<string>();
    public UdpHost UdpController;
    [HideInInspector] public bool UDP_Connection;

    private Vector3 display_offset = new Vector3(0.0f, -2.0f, 0f);

    private void Start()
    {
        foreach (GameObject obj in targets)
        {
            string[] nameParts = obj.name.Split("_");
            if (nameParts.Length >= 3)
            {
                target_list.Add(nameParts[2]);  // e.g., flower_red → gets "red"
            }
            else
            {
                Debug.LogWarning($"❗ Target object '{obj.name}' has unexpected name format.");
            }
        }
    }

    public void Place(GameObject obj)
    {
        if (obj == null) return;

        string[] nameParts = obj.name.Split("_");
        if (nameParts.Length < 2)
        {
            Debug.LogWarning($"❗ Placed object '{obj.name}' has invalid name format.");
            return;
        }

        string flowerType = nameParts[2];

        for (int i = 0; i < target_list.Count; i++)
        {
            if (target_list[i].Contains(flowerType))
            {
                // Snap the flower to its target
                obj.transform.position = targets[i].transform.position;
                obj.transform.SetParent(targets[i].transform);

                Debug.Log($"🌼 Placed '{obj.name}' on '{targets[i].name}'");

                target_list.RemoveAt(i);
                targets.RemoveAt(i);

                if (target_list.Count == 0)
                    VictoryCheck();

                return;
            }
        }

        Debug.Log($"❌ No matching target found for '{flowerType}'");
    }

    private void VictoryCheck()
    {
        Debug.Log("🎉 All flowers placed correctly – Victory!");

        if (floating_text != null)
        {
            Instantiate(floating_text, transform.position - display_offset, Quaternion.identity);
        }

        if (UDP_Connection && UdpController != null)
        {
            UdpController.SendMsg("end");
        }
    }

    public bool CheckObj(GameObject obj)
    {
        if (obj == null) return false;

        string[] nameParts = obj.name.Split("_");
        if (nameParts.Length < 2) return false;

        string flowerType = nameParts[2];

        foreach (string item in target_list)
        {
            if (item.Contains(flowerType)) return true;
        }

        return false;
    }
}
