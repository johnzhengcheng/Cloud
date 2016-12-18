//import java.util.*;
import java.io.*;
//import java.net.*;
//import java.sql.*; //Java»ù´¡°ü£¬°üº¬¸÷ÖÖ±ê×¼Êý¾Ý½á¹¹²Ù×÷

public class cmdThread extends Thread {
	  
	  dataChange link;
	  	public cmdThread()
	  {
	        	
	  }
    
 
    public void run() {
      
    	try
        {
    		String cmd = "sh /root/myCluster/cluster.sh";
			Process child = Runtime.getRuntime().exec(cmd);
			System.out.println(child.exitValue());  //必须打印才真正执行上面的命令
			
			
			
			InputStream child_in = child.getInputStream();
			int c;
			while ((c = child_in.read()) != -1) {
			
				System.out.print((char) c);
			}
			child_in.close();

			System.out.println("The ownership");

        }
        catch (Exception exception)
        {
        
        }
      }

      
    

}



