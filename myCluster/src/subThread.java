import java.util.*;
import java.io.*;
import java.net.*;
import java.sql.*; //Java基础包，包含各种标准数据结构操作

public class subThread extends Thread {
	  
	  dataChange link;
	  	public subThread(dataChange pointer)
	  {
	    link=pointer;
		     	
	  }
    
 
    public void run() {
      
    	try
        {
          // Create a multicast datagram socket for receiving IP
          // multicast packets. Join the multicast group at
          // 230.0.0.1, port 7777.
          MulticastSocket multicastSocket = new MulticastSocket(7777);
          InetAddress inetAddress = InetAddress.getByName("230.0.0.1");
          multicastSocket.joinGroup(inetAddress);
          // Loop forever and receive messages from clients. Print
          // the received messages.
           while (true)
          {
            byte [] arb = new byte [100];
            DatagramPacket datagramPacket = new DatagramPacket(arb, arb.length);
            multicastSocket.setSoTimeout(10000);
            multicastSocket.receive(datagramPacket);
            String strReceived=new String(arb);
            // System.out.println(new String(arb));
            if(strReceived.indexOf("hello")>=0)
                System.out.println(strReceived);
                
          
          }
        }
        catch (Exception exception)
        {
          //exception.printStackTrace();
        link.setValue(0);
        }
      }

      
    

}



